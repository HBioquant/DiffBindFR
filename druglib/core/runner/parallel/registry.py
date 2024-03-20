# Copyright (c) MDLDrugLib. All rights reserved.
import torch.nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from druglib.utils import Registry

MODULE_WRAPPERS = Registry('module wrapper')
MODULE_WRAPPERS.register_module(module = DataParallel)
MODULE_WRAPPERS.register_module(module = DistributedDataParallel)