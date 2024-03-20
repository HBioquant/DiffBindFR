# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, List
import os.path as osp
import time, torch, druglib


from .base_runner import BaseRunner, clean_model_grad
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info


@RUNNERS.register_module()
class EpochBasedRunner(BaseRunner):
    """
    Epoch-based Runner.
    This runner train models epoch by epoch.
    Args:
        enable_avoid_omm (bool, optional): Enable avoid out of memory when training
                Don't support validation time for omm-error skipping.
                Only support EpochBasedRunner. Defaults to False.
    """
    def __init__(
            self,
            *args,
            enable_avoid_omm: bool = False,
            **kwargs,
    ):
        super(EpochBasedRunner, self).__init__(*args, **kwargs)
        if enable_avoid_omm and self.world_size > 1:
            raise RuntimeError('Runner do not support "enbale_avoid_omm" set to True when multi-GPUs Training.')
        if enable_avoid_omm:
            self._avoid_omm_count = 0
        self.enable_avoid_omm = enable_avoid_omm

    def run_iter(
            self,
            data_batch,
            train_mode: bool,
            **kwargs
    ):
        if train_mode:
            outputs = self.model.train_step(
                data_batch,
                self.optimizer,
                **kwargs
            )
        else:
            outputs = self.model.val_step(
                data_batch,
                self.optimizer,
                **kwargs
            )
        if not isinstance(outputs, dict):
            raise TypeError('"model.train_step()" and "model.val_step()" must return a dict')

        if 'log_vars' in outputs:
            self.log_buffer.update(
                vars = outputs['log_vars'], count = outputs['num_samples'],
            )
        self.outputs = outputs

    def train(
            self,
            data_loader,
            **kwargs
    ):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook("before_train_epoch")
        time.sleep(2) #  Prevent possible deadlock during epoch transition
        if self.enable_avoid_omm:
            avoid_omm_count = 0
        for i, data_batch in enumerate(self.data_loader):
            # if i == 4:
            #     break
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook("before_train_iter")
            try:
                self.run_iter(self.data_batch, True, **kwargs)
                self.call_hook("after_train_iter")
            except RuntimeError as e:
                need_avoid_omm = True if 'out of memory' in str(e) else False
                avoid_omm = (need_avoid_omm and self.enable_avoid_omm)
                if not avoid_omm:
                    raise e
                else:
                    self._avoid_omm_count += 1
                    avoid_omm_count += 1 # noqa
                    self.logger.info(f'\033[31;5mOut of memory with {avoid_omm_count} times '
                                     f'in epoch [{self.epoch + 1}] (Total: {self._avoid_omm_count} times), '
                                     f'skipping batch...\033[0m')
                    clean_model_grad(self.model)
            del self.data_batch
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

    @torch.no_grad()
    def val(
            self,
            data_loader,
            **kwargs
    ):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook("before_val_epoch")
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook("before_val_iter")
            self.run_iter(self.data_batch, train_mode = False)
            self.call_hook("after_val_iter")
            del self.data_batch
        self.call_hook("after_val_epoch")

    def run(
            self,
            data_loaders: list,
            workflow: List[tuple],
            **kwargs
    ):
        """
        Start running.

       Args:
           data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
               and validation.
           workflow (list[tuple]): A list of (phase, epochs) to specify the
               running order and epochs. E.g, [('train', 2), ('val', 1)] means
               running 2 epochs for training and 1 epoch for validation,
               iteratively.
           data_loaders must be aligned correctly with workflow
       """
        assert isinstance(data_loaders, list)
        assert druglib.is_list_of(workflow, tuple)
        assert len(workflow) == len(data_loaders)
        # check runner._max_epochs exists
        if self._max_epochs is None:
            raise RuntimeError("'max_epochs must be specified during instantiation'")

        for i, flow in enumerate(workflow):
            mode, epoch = flow
            if mode == 'train':
                self._max_iters = self._max_epochs *  len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)

        self.call_hook('before_run')
        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epoch = flow
                if isinstance(mode, str): # check mode for string type so to call self.train or self.val
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epoch):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1) # wait for some hooks like loggers to finish
        self.call_hook("after_run")


    def save_checkpoint(
            self,
            out_dir:str,
            filename_tmpl:str = 'epoch_{}.pth',
            save_optimizer:bool = True,
            meta:Optional[dict] = None,
            create_symlink: bool = True
    ):
        """
        Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            druglib.symlink(filename, dst_file)
