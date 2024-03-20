# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Optional,
    Tuple,
    List,
    Callable,
    Any,
    Mapping,
    Sequence,
    Union,
)
import numbers
import numpy as np
from functools import partial

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter_sum


def np_repeat(
        arr: np.ndarray,
        n: int
) -> np.ndarray:
    """torch.repeat like repeat function for numpy"""
    if np.isscalar(arr):
        arr = np.array(arr)
    assert isinstance(arr, np.ndarray)
    shape = arr.shape
    if len(shape) == 0:
        arr = np.array([arr])
        shape = arr.shape
    repeat_np = np.tile(arr, (n,) + (1,) * len(shape)).reshape((n,) + shape)
    return repeat_np

def slice_tensor_batch(
        tensor: Tensor,
        batch: Tensor,
        num_graphs: Optional[int] = None,
) -> List[Tensor]:
    """
    Slice the batched graph given the `batch`
    Args:
        tensor: Tensor. Shape (N, ...)
        batch: Tensor. Shape (N, ), every element i  corresponds
            to the (i, ...) of args `tensor`, indicates (i, ...)
            is the node matrix of i-th graph.
        num_graphs: int, optional. The number of graphs batched from
            `tensor`. Defaults to None, inferenced by batch.max() + 1.
    Returns:
        A list of tensor with the length of `num_graphs` or batch.max() + 1.
    """
    if num_graphs is None:
        num_graphs = batch.max().item() + 1
    ls = []
    for i in range(num_graphs):
        mask = (batch == i)
        ls.append(tensor[mask])
    return ls

def slice_tensor_ptr(
        tensor: Tensor,
        ptr: Tensor,
) -> List[Tensor]:
    """
    Slice the batched graph given the `ptr`
    Args:
        tensor: Tensor. Shape (N, ...)
        ptr: Tensor. Shape (M + 1, )
    Returns:
        A list of tensor with the length of M.
    """
    ls = []
    num_graphs = ptr.size(0) - 1
    for i in range(num_graphs):
        ls.append(tensor[ptr[i]:ptr[i + 1]])
    return ls

def ptr_to_batch(
        ptr: Tensor,
) -> Tensor:
    if ptr.numel() == 1:
        if ptr.dim() == 0:
            ptr = ptr.unsqueeze(0)
        assert ptr[0] != 0
        return ptr.new_zeros(ptr[0])

    num_graphs = ptr.size(0) - 1
    num_nodes = torch.diff(ptr)
    return num_nodes_to_batch(
        num_graphs, num_nodes, ptr.device)

def batch_to_ptr(
        batch: Tensor,
):
    bincount = torch.bincount(batch)
    ptr = batch.new_zeros(bincount.size(0) + 1)
    torch.cumsum(bincount, dim = 0, out = ptr[1:])
    return ptr

def ptr_and_batch(
        batch: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
):
    assert (batch is not None or ptr is not None), \
        'At least one from batch or ptr should be set'
    if batch is None and ptr is not None:
        batch = ptr_to_batch(ptr)
    if batch is not None and ptr is None:
        ptr = batch_to_ptr(batch)
    return batch, ptr

def num_nodes_to_batch(
        n_sampels: int,
        num_nodes: Union[int, Tensor],
        device: Union[str, torch.device],
):
    assert isinstance(num_nodes, int) or len(num_nodes) == n_sampels

    if isinstance(num_nodes, Tensor):
        num_nodes = num_nodes.to(device)

    batch_id = torch.arange(n_sampels, device = device)
    return torch.repeat_interleave(batch_id, num_nodes)

def batch_to_num_nodes(
        batch: Tensor,
        batch_size: Optional[int] = None,
):
    ones = torch.ones_like(batch)
    return scatter_sum(ones, batch, dim = 0, dim_size = batch_size)

def batch_tensors(
        tensors: List[Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Batching the unequal-length tensor [tensor1, tensor2, ..., tensorn]
        with [shape(n_1, ...), ..., shape(n_n, ...)] to Tensor with
        shape (n1+...+n_n, ...)
    Args:
        tensors: a list of Tensor.
    Returns:
        Batched Tensor, batch indices and ptr.
    """
    batched = torch.cat(tensors, dim = 0)
    batch = torch.repeat_interleave(
        torch.arange(len(tensors)),
        repeats = torch.LongTensor(t.size(0) for t in tensors)
    ).to(device = batched.device)
    ptr = batch.new_zeros(len(tensors) + 1)
    ptr[1:] = torch.cumsum(torch.tensor([t.size(0) for t in tensors]), dim = 0)
    return batched, batch, ptr

def dict_multimap(
        fn: Callable,
        dicts: List[dict],
):
    """
    Args:
        fn: Map a list of values from dicts with the same key
            to Any.
    """
    ele = dicts[0]
    new = dict()
    for k, v in ele.items():
        vs = [d[k] for d in dicts]
        if isinstance(v, dict):
            new[k] = dict_multimap(fn, vs)
        else:
            new[k] = fn(vs)
    return new

def tree_map(
        fn: Callable,
        tree: Any,
        leaf_type: type,
) -> Any:
    """
    Generalizes the builtin `map` function which only supports flat sequences,
        and allows to apply a function to each "leaf" preserving the overall structure.
    Args:
        fn: Callable.
        tree: nested data structures.
        leaf_type: type.
    Returns:
        Applied fn tree structure.
    """
    if isinstance(tree, Mapping):
        return {k: tree_map(fn, v, leaf_type) for k, v in tree.items()}
    elif isinstance(tree, Sequence) and not isinstance(tree, str):
        return [tree_map(fn, t, leaf_type) for t in tree]
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        # Any not support data type
        # just return it
        return tree

tensor_tree_map = partial(tree_map, leaf_type = Tensor)

def bin_ont_hot(
        tensor: Tensor,
        center_bins: List[float],
) -> Tensor:
    """
    Group a set of values to one single class.
    Using the absolute value to cluster.
    Args:
        tensor: Tensor. (..., N) the last dimension will be
            one-hot encoding.
        center_bins: A list of float-type values. Be Careful to
            design it. The length is M.
    Returns:
        tensor: Tensor. Shape (..., N, M) with M class.
    E.g.:
        >>> center_bins = [2, 4, 7, 9]
        >>> tensor = torch.tensor([1.0, 2.9, 4.0, 19.0])
        >>> bin_ont_hot(tensor, center_bins)
        tensor([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    """
    rs = torch.tensor(center_bins).view((1, ) * len(tensor.shape) + (len(center_bins), )).to(device = tensor.device)
    diff = tensor[..., None] - rs
    arg_min = torch.argmin(torch.abs(diff), dim = -1)
    return F.one_hot(arg_min, num_classes = len(center_bins))

def pts_to_distogram(
        pts: Tensor,
        min_bin: float = 2.3125,
        max_bin: float = 21.6875,
        no_bins: int = 64,
) -> Tensor:
    """
    Args:
        pts: shape (*, N, 3)
    Returns:
        tensor int64, shape (*, N, N)
        for position encoding.
    """
    boundaries = torch.linspace(
        min_bin, max_bin, no_bins - 1,
        device = pts.device)
    dists = torch.sqrt(
        (pts[..., :, None, :] - pts[..., None, :, :]).square().sum(dim = -1)
    )
    return torch.bucketize(dists, boundaries)

def mask2bias(
        mask: Tensor,
        *,
        inf: float = 1e9,
) -> Tensor:
    """
    Convert mask to attention bias
    Args:
        mask: the mask to convert to bias representation
        inf: the floating point number to represent infinity
    Returns:
        bias representation for masking in attention
    """
    return mask.float().sub(1).mul(inf)

def normalize(
        inputs: Tensor,
        normalized_shape: Optional[Union[int, List[int], torch.Size]] = None,
        in_place: bool = False,
) -> Tensor:
    """
    Layer normalization without a module (and weight)

    Args:
        inputs: the input tensor to be normalized
        normalized_shape: the normalized_shape for normalization
        in_place: if to perform the operations in-place

    Returns:
        normalized tensor

    """
    if normalized_shape is None:
        normalized_shape = inputs.shape[-1]
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)

    if in_place:
        # This seems to create small discrepancy in result
        dim = list(range(len(inputs.shape))[-len(normalized_shape):])
        inputs -= inputs.mean(dim = dim, keepdim = True)
        inputs *= torch.rsqrt(inputs.var(dim = dim, keepdim = True) + 1e-5)
        return inputs
    else:
        # F.layer_norm seems a bit faster
        return F.layer_norm(inputs, normalized_shape, None, None, 1e-5)

def robust_normalize(
        x: Tensor,
        dim: int = -1,
        p: Union[int, str] = 2,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    Normalization with a constant small term on the denominator
    Args:
        x: Tensor to normalize
        dim: The dimension along which to perform the normalization
        p: The p in l-p.
    Returns:
        Normalized Tensor
    """
    return x / (x.norm(p = p, dim = dim, keepdim = True).clamp(eps))

def masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: Union[int, Sequence[int], None],
        keepdim: Optional[bool] = False,
        eps: Optional[float] = 1e-9,
) -> torch.Tensor:
    """
    Mean operation with mask
    Args:
        values: the values to take the mean;
        mask: the mask to take the mean with;
        dim: the dimension along which to take the mean;
        keepdim: to keep the dimension;
        eps: the epsilon to compute mean;
    Returns:
        mean result.
    """
    mask = mask.expand(*values.shape)
    values = values.masked_fill(~mask.bool(), 0).sum(dim, keepdim = keepdim)
    denorm = mask.sum(dim, keepdim = keepdim, dtype = values.dtype) + eps
    return values / denorm

