# Copyright (c) MDLDrugLib. All rights reserved.
from itertools import chain
from typing import List, Tuple
from torch.nn.parallel import DataParallel
from .scatter_gather import ScatterInputs, scatter_kwargs


class MDLDataParallel(DataParallel):
    """
    The DataParallel module that supports DataContainer.

    MDLDataParallel has two main differences with PyTorch DataParallel:

    - It supports a custom type :class:`DataContainer` which allows more
      flexible control of input data during both GPU and CPU inference.
    - It implement two more APIs ``train_step()`` and ``val_step()``.

    warning::
        MDLDataParallel only supports single GPU training, if you need to
        train with multiple GPUs, please use MDLDistributedDataParallel
        instead. If you have multiple GPUs and you just want to use
        MDLDataParallel, you can set the environment variable
        ``CUDA_VISIBLE_DEVICES=0`` or instantiate ``MDLDataParallel`` with
        ``device_ids=[0]``.

    Args:
        module (:class:`nn.Module`): Module to be encapsulated.
        device_ids (list[int]): Device IDS of modules to be scattered to.
            Defaults to None when GPU is not available. (default: all devices)
        output_device (str | int): Device ID for output. Defaults to None \equiv device_ids[0]).
        dim (int): Dimension used to scatter the data. Defaults to 0.
    """

    def __init__(self, *args, dim: int = 0, **kwargs):
        super().__init__(*args, dim=dim, **kwargs)
        self.dim = dim

    def scatter(
            self,
            inputs: ScatterInputs,
            kwargs: ScatterInputs,
            device_ids: List[int],
    ) -> Tuple[tuple, tuple]:
        return scatter_kwargs(
            inputs,
            kwargs,
            device_ids,
            self.dim
        )

    def forward(
            self,
            *inputs,
            **kwargs,
    ):
        """
        Override the original forward function.
        Note: This is used for model test and inference.
        The main difference lies in the CPU inference where the data in
        :class:`DataContainers` will still be gathered.
        """
        if not self.device_ids:
            # CPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            # This required single card format in input DataContainer data
            return self.module(*inputs[0], **kwargs[0])
        else:
            return super().forward(*inputs, **kwargs)

    def train_step(
            self,
            *inputs,
            **kwargs,
    ):
        """This function will be called by runner in the train loop"""
        if not self.device_ids:
            # CPU training
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            # This required single card format in input DataContainer data
            return self.module.train_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1, \
            ('MDLDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MDLDistributedDataParallel'
             ' instead.')

        # single card GPU training
        # First, check mixed device case and raise runtime if any
        # because this induce get_input_device in `_function.py` to make mistakes
        for p in chain(self.module.parameters(), self.module.buffers()):
            if p.device != self.src_device_obj:# generally, cuda
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {p.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.train_step(*inputs[0], **kwargs[0])

    def val_step(
            self,
            *inputs,
            **kwargs,
    ):
        """This function will be called by runner in the val loop"""
        if not self.device_ids:
            # CPU training
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            # This required single card format in input DataContainer data
            return self.module.val_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1, \
            ('MDLDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MDLDistributedDataParallel'
             ' instead.')

        # single card GPU training
        # First, check mixed device case and raise runtime if any
        # because this induce get_input_device in `_function.py` to make mistakes
        for p in chain(self.module.parameters(), self.module.buffers()):
            if p.device != self.src_device_obj:# generally, cuda
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {p.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.val_step(*inputs[0], **kwargs[0])