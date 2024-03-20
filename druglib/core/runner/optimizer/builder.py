# Copyright (c) MDLDrugLib. All rights reserved.
import copy, inspect, torch
from typing import Dict, List

from ....utils import Registry, build_from_cfg

OPTIMIZERS = Registry('optimizer')
OPTIMIZERS_BUILDERS = Registry('optimizer builder')

def register_torch_optimizers() -> List[str]:
    torch_optimizers = []
    for module_name in dir(torch.optim):
        # filer to get ['ASGD', 'Adadelta', 'Adagrad',
        # 'Adam', 'AdamW', 'Adamax', 'LBFGS',
        # 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']
        # in torch version == 1.9.0
        if "_" in module_name:
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers

TORCH_OPTIMIZERS = register_torch_optimizers()

def build_optimizer_builder(cfg: Dict):
    return build_from_cfg(
        cfg,
        OPTIMIZERS_BUILDERS,
    )

def builder_optimizer(
        model,
        cfg: Dict,
):
    optimizer_cfg = copy.deepcopy(cfg)
    optimizer_builder = optimizer_cfg.pop(
        "OptimizerBuilder",
        "DefaultOptimizerBuilder",
    )
    paramwise_cfg = optimizer_cfg.pop(
        "paramwise_cfg",
        None
    )
    builder_cfg = {
        "type": optimizer_builder,
        "optimizer_cfg": optimizer_cfg,
        "paramwise_cfg": paramwise_cfg,
    }
    optim_builder = build_optimizer_builder(builder_cfg)
    optimizer = optim_builder(model)
    return optimizer
