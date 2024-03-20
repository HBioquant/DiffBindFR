# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Optional, Union, Sequence,
    Tuple, Dict, Any,
)

import torch
import numpy as np
from torch_sparse import SparseTensor

import druglib
from druglib.data import DataContainer, Data
from ..builder import PIPELINES


def to_tensor(data):
    """
    Convert data object to :obj:`torch.Tensor`.
    Current supported data type includes numpy.ndarray, torch.Tensor
        Sequence, int and float.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not druglib.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'Type {type(data)} is not supported in the current version to convert tensor.')

def coo2sptensor(
        data,
        size: Optional[Tuple[int, int]] = None,
):
    """
    Convert COO format edge_index to :obj:`torch_sparse.SparseTensor`.
    Current supported data type includes numpy.ndarray, torch.Tensor
        Sequence of two Sequence.
    Args:
        data: coo-format egde_index.
        size: Tuple[int, int], optional. SparseTensor input `sparse_sizes`.
            If not specified (None), SparseTensor will simply judge the size by
            `max + 1` algorithm, which is dangerous.
    """
    def _judge_shape(data):
        """input data shape must be (2, x)"""
        return len(data.shape) == 2 and data.shape[0] == 2

    if isinstance(data, np.ndarray) and _judge_shape(data):
        data = to_tensor(data)
    elif isinstance(data, torch.Tensor) and _judge_shape(data):
        pass
    elif isinstance(data, Sequence):
        assert len(data) == 2 and len(data[0]) == len(data[1]), \
            "input sequence data must be the sequence of two same length sequence."
        data = to_tensor(data)
    else:
        raise TypeError(f'Type {type(data)} is not supported in the '
                        'current version to convert coo edge_index to SparseTensor.')
    # finally, get (2, x) shape tensor
    return SparseTensor(row = data[0, :], col = data[1, :], sparse_sizes = size)


@PIPELINES.register_module()
class ToTensor:
    """
    Convert keys of data to :obj:torch.Tensor
    Args；
        keys: Sequence[str]. Keys that need to be converted to Tensor.
    """
    def __init__(self, keys: Sequence[str]):
        self.keys = keys

    def __call__(self, data):
        """
        data: dict-like data type can use 'data[key]' to find value
        """
        for key in self.keys:
            data[key] = to_tensor(data[key])
        return data

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(keys={self.keys})'

@PIPELINES.register_module()
class ToSparseTensor:
    """
        Convert coo-format data to :obj:torch_sparse.SparseTensor
        Args；
            keys: Sequence[str]. Keys that need to be converted to Tensor.
        """

    def __init__(self, keys: Sequence[str]):
        self.keys = keys

    def __call__(self, data):
        """
        data: dict-like data type can use 'data[key]' to find coo-format data
        **Note that data must have attr `{key}_edge_size`
        """
        for key in self.keys:
            edge_size = data.get(f'{key}_edge_size', None)
            data[key] = coo2sptensor(data[key], edge_size)
        return data

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(keys={self.keys})'

@PIPELINES.register_module()
class Transpose:
    """
    Transpose keys of data.
    Args:
        keys: Sequence[str]. Keys that need to be transposed.
        order: Sequence[int]. Order of transpose.
    """
    def __init__(
            self,
            keys: Sequence[str],
            order: Sequence[int],
    ):
        self.keys = keys
        self.order = order

    def __call__(self, data):
        """
        data: dict-like data type can use 'data[key]' to find value
        """
        for key in self.keys:
            data[key] = data[key].transpose(self.order)
        return data

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(keys={self.keys}, order={self.order})'


@PIPELINES.register_module()
class ToDataContainer:
    """
    Wrap data (generally) to :obj:`DataContainer` by given keys
    Args:
        fields: tuple of dict(str, any) or strings. Each field is a dict-like
            `dict(key='edge_index', stack=False, is_graph=True)` or strings,
            defaults to graph data and upload cuda.
        excluded_keys: tuple of str. 'data.data.Data' refused keys.
    """
    def __init__(
            self,
            fields: Tuple[Union[Dict[str, Any], str]] = (
                    dict(key='x', stack=False, is_graph=True),
                    'edge_index',
                    dict(key='img', stack=True, is_graph=False, pad_dims=2),
            ),
            excluded_keys: Tuple[str] = (),
    ):
        self.fields = fields
        self.excluded_keys = excluded_keys

    def __call__(self, data):
        """
        Call function to convert :obj:`DataContainer`
        """
        for k in self.excluded_keys:
            data.pop(k, None)
        for field in self.fields:
            if isinstance(field, dict):
                field = field.copy()
                key = field.pop('key')
            elif druglib.is_str(field):
                key = field
                field = dict(stack = False, is_graph = True)
            else:
                raise TypeError(f'fields must be dict-like or string object, but got {type(field)}.')
            if key == 'metastore':
                continue
            data[key] = DataContainer(data[key], **field)
        return data

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(fields={self.fields})'

@PIPELINES.register_module()
class Collect:
    """
    Collect data from specified keys and meta keys.
    This is usually the second to last stage of the data pipeline.
    """
    def __init__(
            self,
            keys: Sequence[str],
            meta_keys: Sequence[str] = (),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, data):
        """
        Call function to collect keys in data. The `meta_keys` will be
            wrapped into DataContainer with `cpu_only = True` so the last
            stage can recognize it and save it to metastore in :obj:`Data`
            or :obj:`HeteroData`.
        """
        new_data, meta = {}, {}
        for key in self.keys:
            new_data[key] = data[key]
        if 'metastore' in self.meta_keys:
            meta = data.pop('metastore')

        for key in self.meta_keys:
            if key == 'metastore':
                continue
            meta[key] = data[key]
        new_data['metastore'] = meta
        return new_data

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(keys={self.keys}, meta_keys={self.meta_keys})'


@PIPELINES.register_module()
class ToData:
    """Use `data.data.Data` encapsulate data for easy collation."""
    def __call__(self, data) -> Data:
        return Data(**data)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f''
                f')')