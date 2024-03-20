# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Callable, Any, Sequence, Mapping

import torch
from torch.cuda import Stream
from torch_sparse import SparseStorage, SparseTensor

# define a pacther of SparseTensor
def spt_patcher(
        cls: SparseTensor,
) -> SparseTensor:
    """
    Patch the spt_apply base function and two sub-functions to patch SparseTensor.
    User can use this patch function to rebuild :class:SparseTensor data from torch_sparse.SparseTensor
    """
    SparseTensor.spt_apply = spt_apply
    SparseTensor.spt_apply_ = spt_apply_
    SparseTensor.contiguous = contiguous
    SparseTensor.record_stream = record_stream
    return SparseTensor.from_storage(cls.storage)

@classmethod
def spt_apply(
        self: SparseTensor,
        func: Callable,
        *args,
        **kwargs,
):
    """
    Given patched SparseTensor the power of apply more custom function
    """
    return _spt_apply(self, func, *args, **kwargs)

@classmethod
def spt_apply_(
        self: SparseTensor,
        func: Callable,
        *args,
        **kwargs,
):
    """
    Given patched SparseTensor the power of apply more custom function
    """
    return _spt_apply_(self, func, *args, **kwargs)

def contiguous(self: SparseTensor):
    """Apply tensor contiguous function"""
    return self.spt_apply(lambda x: x.contiguous())

def record_stream(self: SparseTensor, stream: Stream):
    """Apply tensor record_stream function for distribution application"""
    return self.spt_apply_(lambda x: x.record_stream(stream))

def _spt_apply(
        cls: SparseTensor,
        func: Callable,
        *args,
        **kwargs,
) -> SparseTensor:
    """Creating a new SparseTensor"""
    self = cls.storage

    row = self._row
    if row is not None:
        row = func(row, *args, **kwargs)

    rowptr = self._rowptr
    if rowptr is not None:
        rowptr = func(rowptr, *args, **kwargs)

    col = func(self._col, *args, **kwargs)

    value = self._value
    if value is not None:
        value = func(value, *args, **kwargs)

    rowcount = self._rowcount
    if rowcount is not None:
        rowcount = func(rowcount, *args, **kwargs)

    colptr = self._colptr
    if colptr is not None:
        colptr = func(colptr, *args, **kwargs)

    colcount = self._colcount
    if colcount is not None:
        colcount = func(colcount, *args, **kwargs)

    csr2csc = self._csr2csc
    if csr2csc is not None:
        csr2csc = func(csr2csc, *args, **kwargs)

    csc2csr = self._csc2csr
    if csc2csr is not None:
        csc2csr = func(csc2csr, *args, **kwargs)

    spstorage = SparseStorage(
        row = row,
        rowptr = rowptr,
        col = col,
        value = value,
        sparse_sizes = self._sparse_sizes,
        rowcount = rowcount,
        colptr = colptr,
        colcount = colcount,
        csr2csc = csr2csc,
        csc2csr = csc2csr,
        is_sorted = True,
        trust_data = True,
    )
    return cls.from_storage(spstorage)

def _spt_apply_(
        cls: SparseTensor,
        func: Callable,
        *args,
        **kwargs,
) -> SparseTensor:
    """In-plane apply function"""
    self = cls.storage

    if self._row is not None:
        func(self._row, *args, **kwargs)

    if self._rowptr is not None:
        func(self._rowptr, *args, **kwargs)

    func(self._col, *args, **kwargs)

    if self._value is not None:
        func(self._value, *args, **kwargs)

    if self._rowcount is not None:
        func(self._rowcount, *args, **kwargs)

    if self._colptr is not None:
        func(self._colptr, *args, **kwargs)

    if self._colcount is not None:
        func(self._colcount, *args, **kwargs)

    if self._csr2csc is not None:
        func(self._csr2csc, *args, **kwargs)

    if self._csc2csr is not None:
        func(self._csc2csr, *args, **kwargs)

    return cls

def recursive_apply_contiguous(
        data: Any
) -> Any:
    """
    Recursive apply function to data, including Tensor, namedtyple, Sequence but not string, Mapping,
        rnn.PackedSequence, SparseTensor.
    Note that SparseTensor does not supports record_stream, contiguous function in PyTorch Geometric.
    We patcher the base :obj:SparseStorage to fix contiguous function.
    """
    if isinstance(data, torch.Tensor):
        return data.contiguous()
    elif isinstance(data, SparseTensor):
        return _spt_apply(data, lambda x: x.contiguous())
    elif isinstance(data, torch.nn.utils.rnn.PackedSequence):
        return data.data.contiguous()
    elif isinstance(data, tuple) and hasattr(data, '_field'):  # namedtuple
        return type(data)(*(recursive_apply_contiguous(d) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [recursive_apply_contiguous(d) for d in data]
    elif isinstance(data, Mapping):
        return {k: recursive_apply_contiguous(v) for k, v in data.items()}
    else:
        try:
            return data.contiguous()
        except:
            return data

def recursive_apply_cudastream(
        data: Any,
        stream: Stream,
) -> Any:
    """
    Recursive apply function to data, including Tensor, namedtyple, Sequence but not string, Mapping,
        rnn.PackedSequence, SparseTensor.
    Note that SparseTensor does not supports record_stream, contiguous function in PyTorch Geometric.
    We patcher the base :obj:SparseStorage to fix record_stream function.
    """
    if isinstance(data, torch.Tensor):
        data.record_stream(stream)
    elif isinstance(data, SparseTensor):
        _spt_apply(data, lambda x: x.record_stream(stream))
    elif isinstance(data, torch.nn.utils.rnn.PackedSequence):
        data.data.record_stream(stream)
    elif isinstance(data, tuple) and hasattr(data, '_field'):  # namedtuple
        for d in data:
            recursive_apply_cudastream(d, stream)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        for d in data:
            recursive_apply_cudastream(d, stream)
    elif isinstance(data, Mapping):
        for v in data.values():
            recursive_apply_cudastream(v, stream)
    else:
        try:
            data.record_stream(stream)
        except:
            return data

def recursive_apply_get_device(
        data: Any,
) -> Any:
    """
    Recursive apply function to data, including Tensor, namedtyple, Sequence but not string, Mapping,
        rnn.PackedSequence, SparseTensor.
    Note that SparseTensor does not supports get_device function in PyTorch Geometric.
    We patcher the base :obj:SparseStorage to fix get_device function.
    """
    if isinstance(data, torch.Tensor):
        return data.get_device()
    elif isinstance(data, SparseTensor):
        return data.storage.col().get_device()
    elif isinstance(data, torch.nn.utils.rnn.PackedSequence):
        return data.data.get_device()
    elif (isinstance(data, tuple) and hasattr(data, '_field')) or \
        isinstance(data, Sequence) and not isinstance(data, str):  # namedtuple or Sequence but not string
        # defualt homogeneous data
        for d in data:
            device = recursive_apply_get_device(d)
            if device != -1:
                return device
        return -1
    elif isinstance(data, Mapping):
        for d in data.values():
            device = recursive_apply_get_device(d)
            if device != -1:
                return device
        return -1
    else:
        try:
            return data.get_device()
        except:
            # unknown data, we think of it as cpu data
            return -1