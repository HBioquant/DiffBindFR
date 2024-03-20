# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Union, Optional, List, Tuple
import torch
import torch.nn as nn

from .registry import MODULE_WRAPPERS
from .data_parallel import MDLDataParallel
from .distributed import MDLDistributedDataParallel

dp = {'cuda': MDLDataParallel, 'cpu': MDLDataParallel}
ddp = {'cuda': MDLDistributedDataParallel}


def is_module_wrapper(
        module: nn.Module
) -> bool:
    """
    Check if a module is a module wrapper.
    The following 3 modules in druglib (and their subclasses) are regarded as
        module wrappers: DataParallel, DistributedDataParallel,
        MDLDistributedDataParallel. You may add you own module wrapper by
        registering it to druglib.runner.parallel.MODULE_WRAPPERS.

    Args:
        module:nn.Module: The module to be checked.

    Returns:
        bool: True if the input module is a module wrapper.
    """
    def is_module_in_wrapper(module, module_wrapper):
        module_wrappers = tuple(module_wrapper.module_dict.values())
        if isinstance(module, module_wrappers):
            return True
        for child in module_wrapper.children.values():
            if is_module_in_wrapper(module, child):
                return True
        return False
    return is_module_in_wrapper(module, MODULE_WRAPPERS)


def is_mlu_available():
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()

def get_device():
    """Returns an available device, cpu, cuda or mlu."""
    is_device_available = {
        'cuda': torch.cuda.is_available(),
        'mlu': is_mlu_available(),
    }
    device_available = [k for k, v in is_device_available.items() if v]
    return device_available[0] if len(device_available) == 1 else 'cpu'

def build_dp(
        model: nn.Module,
        device:str = 'cuda',
        dim: int = 0,
        **kwargs
):
    """build DataParallel module by device type.

    if device is cuda, return a MDLDataParallel model

    Args:
        model (:class:`nn.Module`): model to be parallelized.
        device: str, device type, cuda, cpu or mlu. Defaults to cuda.
        dim: int, dimension used to scatter the data. Defaults to 0.
    Kwargs:
        device_ids: list[int]), device IDS of modules to be scattered to.
            Defaults to None when GPU is not available. (default: all devices)
        output_device: str | int,  device ID for output. Defaults to None \equiv device_ids[0]).
    Returns:
        nn.Module: the model to be parallelized.
    """
    if device == 'cuda':
        model = model.cuda()
    return dp[device](model, dim = dim, **kwargs)

def build_ddp(
        model: nn.Module,
        device: str = 'cuda',
        fp16_compression = False,
        *args,
        **kwargs,
):
    """
    Build DistributedDataParallel module by device type.

   If device is cuda, return a MDLDistributedDataParallel model;

   Args:
       model (:class:`nn.Module`): module to be parallelized.
       device: str, device type, mlu or cuda.
       fp16_compression: bool, False, add fp16 compression hooks to the ddp object.
         See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook

   Returns:
       :class:`nn.Module`: the module to be parallelized

   References:
            [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                    DistributedDataParallel.html
   """
    assert device in ['cuda'], 'Only available for cuda device.'
    if device == 'cuda':
        model = model.cuda()
    DDP = ddp[device](model, *args, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        DDP.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    return DDP