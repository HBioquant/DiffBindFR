# Copyright (c) MDLDrugLib. All rights reserved.
from typing import List, Optional, Union

import torch
from torch import Tensor



def permute_final_dims(
        tensor: Tensor,
        inds: List[int],
):
    """
    e.g.:
        tensor shape (6, 12, 120, 32)
        inds: (2, 0, 1) the last three dims with indices 0, 1, 2
            (ignore the batch dimension)
        >>> tensor.permute([0, -1, -3, -2])
    """
    last_ndim = -1 * len(inds)
    first_dims = list(range(len(tensor.shape[:last_ndim])))
    return tensor.permute(first_dims + [last_ndim + ind for ind in inds])

def flatten_final_dims(
        tensor: Tensor,
        last_ndims: int,
):
    """flatten last N dimensions"""
    return tensor.reshape(tensor.shape[:-last_ndims] + (-1, ))

def binarize(
        tensor: Tensor,
):
    return torch.where(
        tensor > 0,
        torch.ones_like(tensor),
        torch.zeros_like(tensor),
    )

def batched_gather(
        tensor: Tensor,
        indices: Tensor,
        dim: int,
        batch_ndims: int,
) -> Tensor:
    """
    Args:
        tensor: batched tensor with shape (*, N, ...), * represent batched dimension.
        indices: batched indices with the same batched dimension with tensor, shape
            (*, M, ...), M dimension save the indices of tensor's N dimension.
            So output tensor will be (*, M, ...)
        dim: int. indices indexing dimension (e.g. batch_ndims + 1 for tensor's N dimension)
        batch_ndims: int. batched dimension number.
    Returns:
        tensor shape (*, M, ...)
    e.g.::
        tensor shape (A, B, C, D, ..., 37, 3) for all atoms position for protein;
        indices shape (A, B, C, D, ..., 14) for atom14 repr to all atoms 37 repr (save indices \in [0, 36])
        dim: -2, batch_ndims: len(tensor.shape[:-2])
        output: (A, B, C, D, ..., 14, 3)
    """
    # ensure batch dimension consistency
    assert tensor.shape[:batch_ndims] == indices.shape[:batch_ndims]
    # make batch dimension indices
    ranges = []
    for i, b in enumerate(tensor.shape[:batch_ndims]):
        s = torch.arange(b)
        s = s.view(*(*((1, ) * i), -1, *((1, ) * (len(indices.shape) - i - 1))))
        ranges.append(s)
    # make indexing dimension indices
    indices_dims = [slice(None) for _ in range(len(tensor.shape) - batch_ndims)]
    indices_dims[dim - batch_ndims if dim >= 0 else dim] = indices
    # combine batch dimension indices and indixing dimension indices
    ranges.extend(indices_dims)
    return tensor[ranges]

def batched_gather_assign(
        tensor: Tensor,
        indices: Tensor,
        value: Union[Tensor, float, int, bool],
        dim: int,
        batch_ndims: int,
) -> Tensor:
    assert tensor.shape[:batch_ndims] == indices.shape[:batch_ndims]
    ranges = []
    for i, b in enumerate(tensor.shape[:batch_ndims]):
        s = torch.arange(b)
        s = s.view(*(*((1,) * i), -1, *((1,) * (len(indices.shape) - i - 1))))
        ranges.append(s)
    indices_dims = [slice(None) for _ in range(len(tensor.shape) - batch_ndims)]
    indices_dims[dim - batch_ndims if dim >= 0 else dim] = indices
    ranges.extend(indices_dims)
    tensor[ranges] = value
    return tensor

def mask_select(
        src: Tensor,
        dim: int,
        mask: Tensor,
) -> Tensor:
    """
    Returns a new tensor which masks the :obj:`src` tensor along the
    dimension :obj:`dim` according to the boolean mask :obj:`mask`.
    Args:
        src: Tensor. The input tensor.
        dim: int. The dimension in which to mask.
        mask: BoolTensor. The 1-D tensor containing the binary mask to
            index with.
    """
    assert mask.dim() == 1
    assert src.size(dim) == mask.numel()
    dim = dim + src.dim() if dim < 0 else dim
    assert dim >= 0 and dim < src.dim()

    size = [1] * src.dim()
    size[dim] = mask.numel()

    out = src.masked_select(mask.view(size))

    size = list(src.size())
    size[dim] = -1

    return out.view(size)

def index_to_mask(
        index: Tensor,
        size: Optional[int] = None,
) -> Tensor:
    """
    Converts indices to a mask representation.
    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.
    Example:
        >>> index = torch.tensor([1, 3, 5])
        >>> index_to_mask(index)
        tensor([False,  True, False,  True, False,  True])
        >>> index_to_mask(index, size=7)
        tensor([False,  True, False,  True, False,  True, False])
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype = torch.bool)
    mask[index] = True
    return mask


def mask_to_index(
        mask: Tensor,
) -> Tensor:
    """
    Converts a mask to an index representation.
    Args:
        mask: Tensor. The mask.
    Example:
        >>> mask = torch.tensor([False, True, False])
        >>> mask_to_index(mask)
        tensor([1])
    """
    return mask.nonzero(as_tuple = False).view(-1)