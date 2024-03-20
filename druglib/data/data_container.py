# Copyright (c) MDLDrugLib. All rights reserved.
import functools
from typing import Union, Type, Optional

import numpy as np
import torch
from torch import Tensor
import torch_sparse
from torch_sparse import SparseTensor

IndexType = Union[torch.Tensor, np.ndarray, slice, int]

def asser_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # design for DataContainer class, so args[0].data \equiv self.data
        if not isinstance(args[0].data, (Tensor, SparseTensor)):
            raise AttributeError(
                f"{args[0].__class__.__name__} has no attribute "
                f"{func.__name__} for type {args[0].datatype}"
            )

        return func(*args, **kwargs)
    return wrapper

class DataContainer:
    """
    A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array, Tensor or PyG Graph).

    We design `DataContainer` and `MDLDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them, generally suitable
        for image tasks
    - copy to GPU without stacking
        generally suitable for pyg graph data, bbox, object detection label, etc.
    - leave the objects as is and pass it to the model
        e.g. metadata.
    - pad_dims specifies the number of last few dimensions to do padding, e.g. image data
    """
    def __init__(
            self,
            data: Union[Tensor, SparseTensor, np.ndarray],
            stack: bool = False,
            padding_value: int = 0,
            cpu_only: bool = False,
            pad_dims: Optional[int] = None,
            is_graph: bool = True,
    ):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims
        self._is_graph = is_graph

    def __repr__(self) -> str:
        lines = []
        if self.is_tensor:
            lines.append("tensor=" + str(list(self._data.shape)))
        elif self.is_sptensor:
            lines.append("sptensor=" + str(self._data.sizes())[:-1] + f', nnz={self._data.nnz()}]')
        elif self.is_ndarray:
            lines.append("ndarray=" + str(list(self._data.shape)))
        lines.append('type=graph' if self._is_graph else 'type=cv')
        lines.append('device=cpu' if self._cpu_only else 'device=gpu')
        lines.append(f'stack={self._stack}')
        if not self._is_graph:
            lines.append(f'pad value={self._padding_value}')
            lines.append(f'pad dims={self._pad_dims}')
        lines = ", ".join(lines)

        return f'{self.__class__.__name__}({lines})'

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> Union[Tensor, SparseTensor, np.ndarray]:
        return self._data

    @property
    def datatype(self) -> Union[Type]:
        if isinstance(self._data, Tensor):
            return self._data.dtype
        elif isinstance(self._data, SparseTensor):
            return self._data.dtype()
        else:
            return type(self._data)

    @property
    def cpu_only(self) -> bool:
        return self._cpu_only

    @property
    def stack(self) -> bool:
        return self._stack

    @property
    def padding_value(self) -> int:
        return self._padding_value

    @property
    def pad_dims(self) -> Optional[int]:
        return self._pad_dims

    #### define some frequently data-attr-accessible method

    @asser_tensor_type
    def size(self, *args, **kwargs) -> Union[int, torch.Size]:
        return self._data.size(*args, **kwargs)# noqa

    @asser_tensor_type
    def sizes(self) -> Union[torch.Size, list]:
        return self._data.sizes() if self.is_sptensor else self._data.size()

    @asser_tensor_type
    def dim(self) -> int:
        """Prepare for tensor data"""
        return self._data.dim()

    @asser_tensor_type
    def numel(self) -> int:
        return self._data.numel()

    def index(self, index: IndexType) -> 'DataContainer':
        """Index saved tensor and return indexed tensor DataContainer"""
        return DataContainer(
            data = self._data[index],
            stack = self._stack,
            padding_value = self._padding_value,
            cpu_only = self._cpu_only,
            pad_dims = self._pad_dims,
            is_graph = self._is_graph,
        )

    @property
    def max(self) -> Union[Tensor, int, float]:
        return self._data.max()

    @property
    def min(self) -> Union[Tensor, int, float]:
        return self._data.min()

    @property
    def is_tensor(self) -> bool:
        return isinstance(self._data, Tensor)

    @property
    def is_sptensor(self) -> bool:
        return isinstance(self._data, SparseTensor)

    @property
    def is_tensor_or_sptensor(self) -> bool:
        return isinstance(self._data, (Tensor, SparseTensor))

    @property
    def is_ndarray(self) -> bool:
        return isinstance(self._data, np.ndarray)

    @property
    def is_graph(self):
        """
        Is the datacontainer contains graph data such as node data, feature data or graph data,
        Note that this is User well defined attribute.
        """
        return self._is_graph

    def update(self, data: Union[Tensor, SparseTensor, np.ndarray]) -> 'DataContainer':
        """
        Update datacontainer attr `data` with copied other attrs from original one
        Create a new datacontainer
        """
        return DataContainer(
            data = data,
            stack = self._stack,
            padding_value = self._padding_value,
            cpu_only = self._cpu_only,
            pad_dims = self._pad_dims,
            is_graph = self._is_graph,
        )

    def update_(self, data: Union[Tensor, SparseTensor, np.ndarray]) -> 'DataContainer':
        """
        Args:
        In-plane update datacontainer attr `data` with copied other attrs from original one
        """
        self._data = data
        return self

    def __setitem__(
            self,
            index: IndexType,
            value: Union[Tensor, SparseTensor, np.ndarray],
    ):
        self._data[index] = value

    def __getitem__(
            self,
            index: Union[IndexType, 'DataContainer'],
    ):
        if isinstance(index, DataContainer):
            return self._data[index.data]
        return self._data[index]

