# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Union, List, Tuple

import torch
from torch import Tensor
from torch.nn.parallel._functions import Scatter as OrigScatter

from ._functions import Scatter
from druglib.data import DataContainer, BaseData

ScatterInputs = Union[BaseData, Tensor, DataContainer, tuple, list, dict]

def scatter(
        inputs: ScatterInputs,
        target_gpus: List[int],
        dim: int = 0
) -> list:
    """
    Scatter inputs to target gpus.

    The only difference from original :func:`scatter` is to add support for
    :type:`DataContainer`.
    """
    def scatter_map(obj):
        if isinstance(obj, (Tensor, BaseData)):
            if target_gpus != [-1]:
                return OrigScatter.apply(target_gpus, None, dim, obj)
            else:
                # for CPU inference we use self-implement scatter
                return Scatter.forward(target_gpus, obj)
        if isinstance(obj, DataContainer):
            # Then thirdly, section 1
            if obj.cpu_only:
                return obj.data
            else:
                return Scatter.forward(target_gpus, obj.data)
        if isinstance(obj, tuple) and len(obj) > 0:
            # Then, secondly, ("keys", DataContainer) is input,
            # output [("keys", Tensor[Batch,...]), ("keys", Tensor[Batch,...])]
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            # In this part, when target_gpus set to N (len(target_gpus) = N), firstly,
            # "keys":DataContainer([Tensor[Batch,...]] * N)
            # (or "keys":DC(list[list[Batch*] * N]))
            # obj.items() \equiv (("keys", DataContainer), )
            # zip(*map(scatter_map, obj.items())) ->
            # Output: ((("keys", Tensor[Batch,...]), ...(.other keys if keys len > 2)), ...(len(target_gpus) ((...), ...)))
            # apply dict `map(type(obj)...` -> (dict(keys=Tensor[Batch,...], ...(other keys)=...), ...(len(target_gpus) dict(...=...)))
            # so get target_gpus-wise data in the form of dict type.
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        # Then thirdly, section 2
        return [obj for _ in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(
        inputs: ScatterInputs,
        kwargs: ScatterInputs,
        target_gpus: List[int],
        dim: int = 0,
) -> Tuple[tuple, tuple]:
    """Scatter with support for kwargs dictionary."""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        length = len(kwargs) - len(inputs)
        inputs.extend([() for _ in range(length)])
    elif len(kwargs) < len(inputs):
        length = len(inputs) - len(kwargs)
        kwargs.extend([{} for _ in range(length)])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs