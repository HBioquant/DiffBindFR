# Copyright (c) MDLDrugLib. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Optional, Union
from collections import OrderedDict

import torch
from torch import Tensor
import torch.distributed as dist
from druglib.core import BaseModule, auto_fp16
from druglib.data import BaseData


class BaseMLDocker(BaseModule, metaclass = ABCMeta):
    """
    Base class for Machine learning Docking model.
    """
    def __init__(
            self,
            init_cfg: Optional[dict] = None,
    ):
        super(BaseMLDocker, self).__init__(init_cfg)
        self.fp16_enabled = False

    def _parse_losses(self, losses: dict):
        """
        Parse the raw outputs (losses) of the network.
        Args:
            losses: dict. Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple(Tensor, dict): loss and log_vars, loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains all the
                variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors'
                )
        loss = sum(v for k, v in log_vars.items() if 'loss' in k)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device = loss.device)
            dist.all_reduce(log_var_length)
            message = (
                f'rank {dist.get_rank()} ' +
                f'len(log_vars): {len(log_vars)} ' + 'keys: ' +
                ','.join(log_vars.keys())
            )
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
            'loss log variable are different across GPUs:\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(
            self,
            data: Union[BaseData, dict],
            optimizer: Union[torch.optim.Optimizer, dict],
    ) -> dict:
        """
        The iteration step during training

        This method defines an iteration step during training,
            except for the bach-propagation and optimizer updating,
            which are done in an optimizer hook. Note that in some
            cases or models, the whole process including back-progation
            and optimizer updating is also defined in this method, such as GAN.
        Args:
            data: dict or :obj:`BaseData`. Dataset :func:`collate` output.
            optimizer: Union[torch.optim.Optimizer, dict],
                the optimizer of runner is passed to this method.
                This argument is unused and reserved.
        Returns:
            dict:
        """
        assert isinstance(data, (dict, BaseData)), 'input data must be dict or :obj:`BaseData`.'
        if isinstance(data, BaseData):
            data = data.to_dict(decode = True)
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        num_samples = self._infer_batchsize(data)

        outputs = dict(loss = loss, log_vars = log_vars,
                       num_samples = num_samples)

        return outputs

    def _infer_batchsize(self, data: dict):
        """inference num_samples"""
        if 'img_metas' in data:
            num_samples = len(data['img_metas'])
        elif 'ptr' in data:
            num_samples = data['ptr'].size(0) - 1
        elif 'batch' in data:
            num_samples = torch.unique(data['batch']).size(0)
        else:
            raise RuntimeError('Cannnot inference the :key:`num_sampels` '
                               f'from input data with keys {data.keys()}.')
        return num_samples

    def val_step(
            self,
            data: dict,
            optimizer = None,
    ):
        """
        The iteration step during validation.
        This method shares the same signature as :func:`train_step`,
            but used during validation epochs.
        Note that the evaluation after training epochs is not
            implemented with this method, but an evaluation hook.
        """
        assert isinstance(data, (dict, BaseData)), 'input data must be dict or :obj:`BaseData`.'
        if isinstance(data, BaseData):
            data = data.to_dict(decode = True)
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        num_samples = self._infer_batchsize(data)

        outputs = dict(loss = loss, log_vars = log_vars,
                       num_samples = num_samples)

        return outputs

    @auto_fp16(apply_to = None)
    def forward(
            self,
            *args,
            mode:str = "train",
            **kwargs,
    ):
        if not isinstance(mode, str):
            raise TypeError(f"kwargs `mode` must be string, but got {type(mode)}")
        if mode not in ["train", "test"]:
            raise ValueError(f"kwargs `mode` must be 'train' or 'test', but got '{mode}'")

        return getattr(self, f"forward_{mode}")(*args, **kwargs)

    @abstractmethod
    def forward_train(
            self,
            *args,
            **kwargs,
    ):
        pass

    @abstractmethod
    def forward_test(
            self,
            *args,
            **kwargs,
    ):
        pass