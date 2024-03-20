# Copyright (c) MDLDrugLib. All rights reserved.
import warnings, torch
from .version_utils import digit_version
from .parrots_wrapper import TORCH_VERSION

def is_jit_tracing() -> bool:
    if TORCH_VERSION != 'parrots' and digit_version(TORCH_VERSION) >= digit_version('1.6.0'):
        on_trace = torch.jit.is_tracing()
        # In PyTorch 1.6, torch,jit.is_tracing has a bug.
        # Refers ti https://github.com/pytorch/pytorch/issues/42448
        if isinstance(on_trace, bool):
            return on_trace
        else:
            return torch._C._is_tracing()
    else:
        warnings.warn(
            'torch.jit.is_tracing is only supported after v1.6.0. '
            'Therefore is_tracing returns False automatically. Please '
            'set on_trace manually if you are using trace.', UserWarning
        )
        return False