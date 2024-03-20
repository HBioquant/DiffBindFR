# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, Union
import numpy as np
import torch
from torch import Tensor


def check_inf_nan_np(
        array: Union[np.ndarray],
) -> bool:
    return not (np.isnan(array).any() or np.isinf(array).any())

def check_inf_nan_torch(
        tensor: Tensor,
) -> bool:
    return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())


