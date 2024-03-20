# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Union,
    List,
    Tuple,
    Optional
)
import numpy as np
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter, scatter_add

from druglib.data.typing import Adj, OptTensor, PairTensor
from .tensor_extension import index_to_mask


def get_complete_graph(
        num_nodes_per_graph: Union[Tensor, List[int], np.ndarray]
) -> Tuple[Tensor, Tensor]:
    """
    Get the complete graph from the number of nodes per graph.
    Args:
        num_nodes_per_graph: Tensor, list of int or array are allowed.
            The length of `num_nodes_per_graph` is the number of graph.
            Shape (N, )
    Returns:
        edge_index: (2, N_1 + N_2 + ... + N_{B-1}), where N_i is the number
            of egdes of the i-th graph. Row-permuted.
        num_edges:  (B, ), number of edges per graph.
    E.g.:
        >>> get_complete_graph(torch.tensor([4, 2, 3]))
        (tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8],
                 [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4, 7, 8, 6, 8, 6, 7]]),
         tensor([12,  2,  6]))
        # build the complete graph from `batch`
        >>> batch = torch.tensor([0,0,0,0,1,1,2,2,2], dtype = torch.long)
        >>> from torch_scatter import scatter_add
        >>> num_nodes_per_graph = scatter_add(torch.ones_like(batch), index=batch, dim=0)
        >>> get_complete_graph(num_nodes_per_graph)
        (tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6, 6, 7, 7, 8, 8],
                 [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4, 7, 8, 6, 8, 6, 7]]),
         tensor([12,  2,  6]))
    """
    if not isinstance(num_nodes_per_graph, Tensor):
        num_nodes_per_graph = torch.Tensor(num_nodes_per_graph)

    # the number of node pairs per graph
    np_sqr = (num_nodes_per_graph ** 2).long()
    # the total number of nodes
    np_pairs = torch.sum(np_sqr)
    # the row(/col) number per graph
    num_row = torch.repeat_interleave(num_nodes_per_graph, np_sqr) # num_row == num_col

    # the batch edge_index incr
    index_incr = torch.cumsum(num_nodes_per_graph, dim = 0) - num_nodes_per_graph
    index_inc_offset = torch.repeat_interleave(index_incr, np_sqr)

    # the node idx  (row_id * num_col + col_id) per graph
    num_nodes_per_graph_offset = torch.cumsum(np_sqr, dim=0) - np_sqr
    num_nodes_per_graph_offset = torch.repeat_interleave(
        num_nodes_per_graph_offset,
        np_sqr)# graph-level repeat to node-level
    node_id_per_graph = torch.arange(
        np_pairs,
        device = np_pairs.device) - num_nodes_per_graph_offset

    # get (2, num_edges) edge_index
    src = (torch.div(node_id_per_graph, num_row, rounding_mode = 'trunc')).long() + index_inc_offset
    dst = (node_id_per_graph % num_row).long() + index_inc_offset
    edge_index = torch.stack([src, dst])

    # delete self-loop
    mask = (src != dst)
    edge_index = edge_index[:, mask]

    num_edges = np_sqr - num_nodes_per_graph

    return edge_index.long(), num_edges.long()

def get_complete_bipartite_graph(
        num_nodes_per_src_graph: Union[Tensor, List[int], np.ndarray],
        num_nodes_per_dst_graph: Union[Tensor, List[int], np.ndarray],
):
    """
    Get the complete bipartite graph from the number of nodes per graph.
    Args:
        num_nodes_per_src_graph: Tensor, list of int or array are allowed.
            The length of `num_nodes_per_graph` is the number of graph.
            Shape (N, )
        num_nodes_per_dst_graph: Tensor, the same as num_nodes_per_src_graph
            in shape.
    Returns:
        edge_index: (2, N_1 + N_2 + ... + N_{B-1}), where N_i is the number of
            edges of the i-th bipartite graph. Row-permuted.
        num_edges:  (B, ), number of edges per graph.
    E.g.:
        >>> get_complete_bipartite_graph(
                        torch.tensor([2, 3, 2]),
                        torch.tensor([4, 2, 5]))
        (tensor([[ 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  5,  5,
                5,  6,  6,  6,  6,  6],
                [ 0,  1,  2,  3,  0,  1,  2,  3,  4,  5,  4,  5,  4,  5,  6,  7,  8,  9,
                10,  6,  7,  8,  9, 10]]),
        tensor([ 8,  6, 10]))
    """
    if not isinstance(num_nodes_per_src_graph, Tensor):
        num_nodes_per_src_graph = torch.Tensor(num_nodes_per_src_graph)
    if not isinstance(num_nodes_per_dst_graph, Tensor):
        num_nodes_per_dst_graph = torch.Tensor(num_nodes_per_dst_graph)

    # the number of node pairs per graph
    np_sqr = (num_nodes_per_src_graph * num_nodes_per_dst_graph).long()
    # the total number of nodes
    np_pairs = torch.sum(np_sqr)
    # the row(/col) number per graph
    num_dst = torch.repeat_interleave(num_nodes_per_dst_graph, np_sqr)

    # the batch edge_index incr
    index_incr_src = torch.cumsum(num_nodes_per_src_graph, dim = 0) - num_nodes_per_src_graph
    index_inc_offset_src = torch.repeat_interleave(index_incr_src, np_sqr)

    index_incr_dst = torch.cumsum(num_nodes_per_dst_graph, dim = 0) - num_nodes_per_dst_graph
    index_inc_offset_dst = torch.repeat_interleave(index_incr_dst, np_sqr)

    # the node idx  (row_id * num_col + col_id) per graph
    num_nodes_per_graph_offset = torch.cumsum(np_sqr, dim = 0) - np_sqr
    num_nodes_per_graph_offset = torch.repeat_interleave(
        num_nodes_per_graph_offset,
        np_sqr)  # graph-level repeat to node-level
    node_id_per_graph = torch.arange(
        np_pairs,
        device = np_pairs.device) - num_nodes_per_graph_offset

    # get (2, num_edges) edge_index
    src = (torch.div(node_id_per_graph, num_dst, rounding_mode = 'trunc')).long() + index_inc_offset_src
    dst = (node_id_per_graph % num_dst).long() + index_inc_offset_dst
    edge_index = torch.stack([src, dst])

    return edge_index.long(), np_sqr.long()

def get_complete_subgraph(
        node_id: Union[Tensor, List[int], np.ndarray]
):
    """
    Args `node_id` is the subgraph id from the graph,
        and this function will build the complete graph
        for the subgraph with orignal edge index (No relabel)
    Args:
        node_id: Tensor, list of int or array are allowed.
        Such as [2, 4, 5] represents the 3rd, 5th, 6th node in a graph
        And then we expect to get the complete graph
           [[2, 2, 4, 4, 5, 5]
            [4, 5, 2, 5, 2, 4]]
    Returns:
        The complete subgraph edge_index with the same data type as `node_id`.
    E.g.:
        >>> node_id = [2,3,5,7,9]
        >>> get_complete_subgraph(node_id)
        [[2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9],
         [3, 5, 7, 9, 2, 5, 7, 9, 2, 3, 7, 9, 2, 3, 5, 9, 2, 3, 5, 7]]
    """
    if isinstance(node_id, (Tensor, np.ndarray)):
        assert len(node_id.shape) == 1
        if isinstance(node_id, Tensor):
            node_id = node_id.clone().cpu()
        node_id = node_id.tolist()

    node_index, _ = get_complete_graph([len(node_id)])
    row, col = node_index
    row = np.array(node_id)[row]
    col = np.array(node_id)[col]
    edge_index = np.stack([row, col], axis=0)
    if isinstance(node_id, Tensor):
        return torch.from_numpy(edge_index).to(device = node_id.device)
    if isinstance(node_id, np.ndarray):
        return edge_index
    if isinstance(node_id, list):
        return edge_index.tolist()

def remove_duplicate_edges(
        edge_index: Adj,
        return_inverse: bool = False,
        return_ind: bool = False,
) -> Tuple[Adj, OptTensor, OptTensor]:
    """
    Remove the duplicated edge in the graph.
    Compared to the coalesce algorithm, it ignore
        the edge_attr and save computational time.
    Args:
        edge_index: Tensor or SparseTensor. Shape(2, num_edge).
        Note that the dim will be fixed to `1`. See PyTorch.unique for details.
        return_inverse: bool. See PyTorch.unique for details.
            Defaults to False.
            The main function is that output the indices to restore
                the original input edge_index.
        return_ind: bool. The main function is that output the indices from
            the input edge_index.
    Returns:
        unique, inverse, ori_ind:
        if input is `x`, then unique[inverse] == x
                              x[ori_ind] == unique.

    E.g.:
        >>> x = torch.tensor([
            [1, 2, 1, 4, 5],
            [2, 3, 2, 3, 1]])
        >>> a, _, b = remove_duplicate_edges(x, return_ind=True)
        (tensor([[1, 2, 4, 5],
                [2, 3, 3, 1]]),
        tensor([0, 1, 3, 4]))
        >>> x[:,b]
        tensor([[1, 2, 4, 5],
                [2, 3, 3, 1]])
        >>> a, b, _ = remove_duplicate_edges(x, return_inverse=True)
        (tensor([[1, 2, 4, 5],
                [2, 3, 3, 1]]),
        tensor([0, 1, 0, 2, 3]))
        >>> a[:, b]
        tensor([[1, 2, 1, 4, 5],
                [2, 3, 2, 3, 1]])
    """
    # SparseTensor will be coalesced
    if isinstance(edge_index, SparseTensor):
        return edge_index, None, None

    if isinstance(edge_index, Tensor):
        assert len(edge_index.shape) == 2 and edge_index.shape[0] == 2, \
            "This function is written specifically for `edge_index` with shape (2, num_edge)."

    unique, inverse = torch.unique(
        edge_index, sorted = True, return_inverse=True, dim = 1)
    if not (return_ind or return_ind):
        return unique

    if return_inverse and not return_ind:
        return unique, inverse, None

    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse_flip, perm = inverse.flip([0]), perm.flip([0])
    ori_ind = inverse_flip.new_empty(unique.size(1)).scatter_(0, inverse_flip, perm)
    if return_ind and not return_inverse:
        return unique, None, ori_ind

    return unique, inverse, ori_ind

def maybe_batch(
        root_tensor: Tensor,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        root_tensor_dim: int = 0,
) -> Tuple[Tensor, Tensor, int]:
    if batch is None:
        batch = root_tensor.new_zeros(root_tensor.size(root_tensor_dim)).long()
    if batch_size is None:
        batch_size = int(batch.max().item() + 1)
    if num_nodes is None:
        ones = torch.ones_like(batch, dtype = torch.int64)
        num_nodes = scatter_add(ones, batch, dim = 0, dim_size = batch_size)
    return batch, num_nodes, batch_size


############### Copy from pytorch_geometric
############### The old version of PyG has not the :func:`maybe_num_nodes`
@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (Tensor, Optional[int]) -> int
    pass


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (SparseTensor, Optional[int]) -> int
    pass


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def remove_self_loops(
        edge_index: Tensor,
        edge_attr: OptTensor = None,
) -> Tuple[Tensor, OptTensor]:
    """
    Removes every self-loop in the graph given by :attr:`edge_index`.
    Args:
        edge_index: LongTensor. The edge indices.
        edge_attr: Tensor, optional. Edge weights or multi-dimensional
            edge features. Defaults to :obj:`None`.
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def to_dense_adj(
        edge_index: Tensor,
        batch: OptTensor = None,
        edge_attr: OptTensor = None,
        max_num_nodes = None,
        batch_size: Optional[int] = None,
        full_value: float = 0.,
) -> Tensor:
    """
    Convert batched sparse adjacency matrics given by edge
        indices and edge attributes to a single dense batched
        adjacency matrix.
    Args:
        edge_index: LongTensor with shape (2, N). The edge indices.
        batch: LongTensor, optional. Batch vector. Defaults to None, a single samples.
        edge_attr: Tensor, optional. Edge weights or multi-dimensional edge features.
            Defaults to None.
        max_num_nodes: int, optional. The size of the output node dimension.
            Defaults to None.
    Returns:
        dense adjacency matrices with shape (B, N_max, f_dim).
    E.g.:
        >>> edge_index = torch.tensor([[0, 0, 1, 2, 3],
        ...                            [0, 1, 0, 3, 0]])
        >>> batch = torch.tensor([0, 0, 1, 1])
        >>> to_dense_adj(edge_index, batch)
        tensor([[[1., 1.],
                [1., 0.]],
                [[0., 1.],
                [1., 0.]]])
        >>> to_dense_adj(edge_index, batch, max_num_nodes=4)
        tensor([[[1., 1., 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]],
                [[0., 1., 0., 0.],
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]]])
        >>> edge_attr = torch.Tensor([1, 2, 3, 4, 5])
        >>> to_dense_adj(edge_index, batch, edge_attr)
        tensor([[[1., 2.],
                [3., 0.]],
                [[0., 4.],
                [5., 0.]]])
    """
    if batch is None:
        batch = edge_index.new_zeros(maybe_num_nodes(edge_index))

    if batch_size is None:
        batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1

    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_add(one, batch, dim = 0, dim_size = batch_size)
    ptr = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim = 0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - ptr[batch][edge_index[0]]
    idx2 = edge_index[1] - ptr[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()
    elif ((idx1.numel() > 0 and idx1.max() >= max_num_nodes)
          or (idx2.numel() > 0 and idx2.max() >= max_num_nodes)):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device = edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    flattened_size = batch_size * max_num_nodes * max_num_nodes

    adj = torch.full(
        size, fill_value = full_value,
        dtype = edge_attr.dtype,
        device = edge_index.device)
    adj = adj.view([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    scatter_add(edge_attr, idx, dim = 0, out = adj)
    adj = adj.view(size)

    return adj


def global_add_pool(
        x: Tensor,
        batch: Optional[Tensor],
        size: Optional[int] = None,
) -> Tensor:
    """
    Returns batch-wise graph-level-outputs by adding node features
        across the node dimension.
    Args:
        x: Node feature matrix
        batch: LongTensor, optional. Batch vector
        size: int, optional. Batch-size.
    """
    if batch is None:
        return x.sum(dim = -2, keepdim = x.dim() == 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim = -2, dim_size = size, reduce = 'add')


def global_mean_pool(
        x: Tensor,
        batch: Optional[Tensor],
        size: Optional[int] = None,
) -> Tensor:
    """
    Returns batch-wise graph-level-outputs by averaging node features
        across the node dimension.
    Args:
        x: Node feature matrix
        batch: LongTensor, optional. Batch vector
        size: int, optional. Batch-size.
    """
    if batch is None:
        return x.mean(dim = -2, keepdim = x.dim() == 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim = -2, dim_size = size, reduce = 'mean')


def global_max_pool(
        x: Tensor,
        batch: Optional[Tensor],
        size: Optional[int] = None,
) -> Tensor:
    """
    Returns batch-wise graph-level-outputs by taking the channel-wise
        maximum across the node dimension.
    Args:
        x: Node feature matrix
        batch: LongTensor, optional. Batch vector
        size: int, optional. Batch-size.
    """
    if batch is None:
        return x.max(dim = -2, keepdim = x.dim() == 2)[0]
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim = -2, dim_size = size, reduce = 'max')

def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, OptTensor]]:
    """
    Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.
    Args:
        subset: LongTensor, BoolTensor or [int]. The nodes to keep.
        edge_index: LongTensor. The edge indices.
        edge_attr: Tensor, optional. Edge weights or multi-dimensional
            edge features. Defaults to :obj:`None`.
        relabel_nodes: bool, optional. If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. Defaults to :obj:`False`.
        num_nodes: int, optional. The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. Defaults to :obj:`None`.
        return_edge_mask: bool, optional. If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            Defaults to :obj:`False`.
    Examples:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        >>> subset = torch.tensor([3, 4, 5])
        >>> subgraph(subset, edge_index, edge_attr)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]))
        >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]),
        tensor([False, False, False, False, False, False,  True,
                True,  True,  True,  False, False]))
    """

    device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype = torch.long, device = device)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        num_nodes = subset.size(0)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        subset = index_to_mask(subset, size = num_nodes)

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx = torch.zeros(node_mask.size(0), dtype = torch.long,
                               device = device)
        node_idx[subset] = torch.arange(subset.sum().item(), device = device)
        edge_index = node_idx[edge_index]

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


def bipartite_subgraph(
    subset: Union[PairTensor, Tuple[List[int], List[int]]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    size: Optional[Tuple[int, int]] = None,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, OptTensor]]:
    """
    Returns the induced subgraph of the bipartite graph
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.
    Args:
        subset: Tuple[Tensor, Tensor] or tuple([int],[int]). The nodes
            to keep.
        edge_index: LongTensor. The edge indices.
        edge_attr: Tensor, optional. Edge weights or multi-dimensional
            edge features. Defaults to :obj:`None`.
        relabel_nodes: bool, optional. If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. Defaults to :obj:`False`.
        size: tuple, optional. The number of nodes. Defaults to :obj:`None`.
        return_edge_mask: bool, optional. If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            Defaults to :obj:`False`.
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    Examples:
        >>> edge_index = torch.tensor([[0, 5, 2, 3, 3, 4, 4, 3, 5, 5, 6],
        ...                            [0, 0, 3, 2, 0, 0, 2, 1, 2, 3, 1]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        >>> subset = (torch.tensor([2, 3, 5]), torch.tensor([2, 3]))
        >>> bipartite_subgraph(subset, edge_index, edge_attr)
        (tensor([[2, 3, 5, 5],
                [3, 2, 2, 3]]),
        tensor([ 3,  4,  9, 10]))
        >>> bipartite_subgraph(subset, edge_index, edge_attr,
        ...                    return_edge_mask=True)
        (tensor([[2, 3, 5, 5],
                [3, 2, 2, 3]]),
        tensor([ 3,  4,  9, 10]),
        tensor([False, False,  True,  True, False, False, False, False,
                True,  True,  False]))
    """

    device = edge_index.device

    src_subset, dst_subset = subset
    if not isinstance(src_subset, Tensor):
        src_subset = torch.tensor(src_subset, dtype = torch.long, device = device)
    if not isinstance(dst_subset, Tensor):
        dst_subset = torch.tensor(dst_subset, dtype = torch.long, device = device)

    if src_subset.dtype != torch.bool:
        src_size = int(edge_index[0].max()) + 1 if size is None else size[0]
        src_subset = index_to_mask(src_subset, size = src_size)
    if dst_subset.dtype != torch.bool:
        dst_size = int(edge_index[1].max()) + 1 if size is None else size[1]
        dst_subset = index_to_mask(dst_subset, size = dst_size)

    # node_mask = subset
    edge_mask = src_subset[edge_index[0]] & dst_subset[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx_i = edge_index.new_zeros(src_subset.size(0))
        node_idx_j = edge_index.new_zeros(dst_subset.size(0))
        node_idx_i[src_subset] = torch.arange(
            int(src_subset.sum()), device = node_idx_i.device)
        node_idx_j[dst_subset] = torch.arange(
            int(dst_subset.sum()), device = node_idx_j.device)
        edge_index = torch.stack([
            node_idx_i[edge_index[0]],
            node_idx_j[edge_index[1]],
        ], dim = 0)

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


def k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.
    The :attr:`flow` argument denotes the direction of edges for finding
    :math:`k`-hop neighbors. If set to :obj:`"source_to_target"`, then the
    method will find all neighbors that point to the initial set of seed nodes
    in :attr:`node_idx.`
    This mimics the natural flow of message passing in Graph Neural Networks.
    The method returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx: int, list, tuple or :obj:`torch.Tensor`. The central seed
            node(s).
        num_hops: int. The number of hops :math:`k`.
        edge_index: LongTensor. The edge indices.
        relabel_nodes: bool, optional. If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. Defaults to :obj:`False`.
        num_nodes: int, optional. The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. Defaults to :obj:`None`.
        flow: str, optional. The flow direction of :math:`k`-hop aggregation
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            Defaults to :obj:`"source_to_target"`.
        directed: bool, optional. If set to :obj:`False`, will include all
            edges between all sampled nodes. Defaults to :obj:`True`.
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    Examples:
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
        ...                            [2, 2, 4, 4, 6, 6]])
        >>> # Center node 6, 2-hops
        >>> subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        ...     6, 2, edge_index, relabel_nodes=True)
        >>> subset
        tensor([2, 3, 4, 5, 6])
        >>> edge_index
        tensor([[0, 1, 2, 3],
                [2, 2, 4, 4]])
        >>> mapping
        tensor([4])
        >>> edge_mask
        tensor([False, False,  True,  True,  True,  True])
        >>> subset[mapping]
        tensor([6])
        >>> edge_index = torch.tensor([[1, 2, 4, 5],
        ...                            [0, 1, 5, 6]])
        >>> (subset, edge_index,
        ...  mapping, edge_mask) = k_hop_subgraph([0, 6], 2,
        ...                                       edge_index,
        ...                                       relabel_nodes=True)
        >>> subset
        tensor([0, 1, 2, 4, 5, 6])
        >>> edge_index
        tensor([[1, 2, 3, 4],
                [0, 1, 4, 5]])
        >>> mapping
        tensor([0, 5])
        >>> edge_mask
        tensor([True, True, True, True])
        >>> subset[mapping]
        tensor([0, 6])
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype = torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype = torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device = row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out = edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse = True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device = row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask
