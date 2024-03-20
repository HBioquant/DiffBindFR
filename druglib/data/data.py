# Copyright (c) MDLDrugLib. All rights reserved.
# Modified from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/data.py
import copy
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils import subgraph

from .typing import (
    EdgeTensorType,
    EdgeType,
    NodeType,
    OptInput,
    CVWORLD,
    CVKEYS
)
from .feature_store import (
    FeatureStore,
    FeatureTensorType,
    TensorAttr,
    _field_status,
)
from .graph_store import (
    EDGE_LAYOUT_TO_ATTR_NAME,
    EdgeAttr,
    EdgeLayout,
    GraphStore,
    adj_type_to_edge_tensor_type,
    edge_tensor_type_to_adj_type,
)
from .storage import (
    BaseStorage,
    EdgeStorage,
    NodeStorage,
    CVStorage,
    GlobalStorage,
)
from .data_container import DataContainer
from druglib.utils import color

class BaseData(object):
    def __getattr__(self, key: str) -> Any:
        raise NotImplementedError

    def __setattr__(self, key: str, value: Any):
        raise NotImplementedError

    def __delattr__(self, key: str):
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError

    def __delitem__(self, key: str):
        raise NotImplementedError

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memo):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def stores_as(self, data: 'BaseData'):
        """Hetero data interface. In Data, it is a placeholder"""
        raise NotImplementedError

    @property
    def stores(self) -> List[BaseStorage]:
        """
        Data holds one-length list of BaseStorage
        Hetero data holds a list of multiple BaseStorage
        """
        raise NotImplementedError

    @property
    def node_stores(self) -> List[NodeStorage]:
        """Hetero data interface"""
        raise NotImplementedError

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        """Hetero data interface"""
        raise NotImplementedError

    @property
    def cv_stores(self) -> List[CVStorage]:
        """Hetero datd cv data store interface"""
        raise NotImplementedError

    @property
    def meta_stores(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def to_dict(self, decode: bool) -> Dict[str, Any]:
        """Returns a dictionary of stored key/value pairs."""
        raise NotImplementedError

    def to_namedtuple(self, decode: bool) -> NamedTuple:
        """Returns a :obj:`NamedTuple` of stored key/value pairs."""
        raise NotImplementedError

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        """
        Returns the dimension for which the value :obj:`value` of the
            attribute :obj:`key` will get concatenated when creating
            mini-batches using :class:`torch_geometric.loader.DataLoader`.
        .. note::
            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        """
        Returns the incremental count to cumulatively increase the value
            :obj:`value` of the attribute :obj:`key` when creating mini-batches
            using :class:`torch_geometric.loader.DataLoader`.
        .. note::
            This method is for internal use only, and should only be overridden
            in case the mini-batch creation process is corrupted for a specific
            attribute.
        """
        raise NotImplementedError


    ###########################################################################

    @property
    def keys(self) -> List[str]:
        """Returns a list of all graph attribute names."""
        out = []
        for store in self.stores:
            out += list(store.keys())
        return list(set(out))

    def __len__(self) -> int:
        """Returns the number of graph attributes."""
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        """
        Returns :obj:`True` if the attribute :obj:`key` is present in the data.
        """
        return key in self.keys

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__

    def __setstate__(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.__dict__[key] = value

    @property
    def num_nodes(self) -> Optional[int]:
        """
        Returns the number of nodes in the graph.
        .. note::
            The number of nodes in the data object is automatically inferred
            in case node-level attributes are present, *e.g.*, :obj:`data.x`.
            In some cases, however, a graph may only be given without any
            node-level attributes.
            PyG then *guesses* the number of nodes according to
            :obj:`edge_index.max().item() + 1`.
            However, in case there exists isolated nodes, this number does not
            have to be correct which can result in unexpected behaviour.
            Thus, we recommend to set the number of nodes in your data object
            explicitly via :obj:`data.num_nodes = ...`.
            You will be given a warning that requests you to do so.
        """
        try:
            return sum([v.num_nodes for v in self.node_stores])
        except TypeError:
            return None

    def numel(self) -> int:
        """Count the :cls:`DataContainer` dict tensor numel summation"""
        numel = 0
        for store in self.stores:
            for v in store.values():
                numel += v.numel()
        return numel

    def size(
        self,
        dim: Optional[int] = None
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        """Returns the size of the adjacency matrix induced by the graph."""
        size = (self.num_nodes, self.num_nodes)
        return size if dim is None else size[dim]

    @property
    def num_edges(self) -> int:
        """
        Returns the number of edges in the graph.
        For undirected graphs, this will return the number of bi-directional
            edges, which is double the amount of unique edges.
        """
        return sum([v.num_edges for v in self.edge_stores])

    def is_coalesced(self) -> bool:
        """
        Returns :obj:`True` if edge indices :obj:`edge_index` are sorted
            and do not contain duplicate entries.
        """
        return all([store.is_coalesced() for store in self.edge_stores])

    def coalesce(self):
        """
        Sorts and removes duplicated entries from edge indices
            :obj:`edge_index`.
        """
        for store in self.edge_stores:
            store.coalesce()
        return self

    def has_isolated_nodes(self) -> bool:
        """Returns :obj:`True` if the graph contains isolated nodes."""
        return any([store.has_isolated_nodes() for store in self.edge_stores])

    def has_self_loops(self) -> bool:
        """Returns :obj:`True` if the graph contains self-loops."""
        return any([store.has_self_loops() for store in self.edge_stores])

    def is_undirected(self) -> bool:
        """Returns :obj:`True` if graph edges are undirected."""
        return all([store.is_undirected() for store in self.edge_stores])

    def is_directed(self) -> bool:
        """Returns :obj:`True` if graph edges are directed."""
        return not self.is_undirected()

    def apply_(self, func: Callable, *args: List[str]):
        """
        Applies the in-place function :obj:`func`, either to all attributes
            or only the ones given in :obj:`*args`.
        """
        for store in self.stores:
            store.apply_(func, *args)
        return self

    def apply(self, func: Callable, *args: List[str]):
        """
        Applies the function :obj:`func`, either to all attributes or only
            the ones given in :obj:`*args`.
        """
        for store in self.stores:
            store.apply(func, *args)
        return self

    def clone(self, *args: List[str]):
        """
        Performs cloning of tensors, either for all attributes or only the
            ones given in :obj:`*args`.
        """
        return copy.copy(self).apply(lambda x: x.clone(), *args)

    def contiguous(self, *args: List[str]):
        """
        Ensures a contiguous memory layout, either for all attributes or
            only the ones given in :obj:`*args`.
        """
        for store in self.stores:
            store.contiguous(*args)
        return self

    def to(
            self, device: Union[int, str], *args: List[str],
            non_blocking: bool = False,
    ):
        """
        Performs tensor device conversion, either for all attributes or
            only the ones given in :obj:`*args`.
        """
        for store in self.stores:
            store.to(device = device, *args, non_blocking = non_blocking)
        return self

    def cpu(self, *args: List[str]):
        """
        Copies attributes to CPU memory, either for all attributes or only
            the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.cpu(), *args)

    def cuda(
            self, device: Optional[Union[int, str]] = None, *args: List[str],
            non_blocking: bool = False,
    ):
        """
        Copies attributes to CUDA memory, either for all attributes or only
            the ones given in :obj:`*args`.
        """
        # Some PyTorch tensor like objects require a default value for `cuda`:
        device = 'cuda' if device is None else device
        for store in self.stores:
            store.cuda(device = device, *args, non_blocking = non_blocking)
        return self

    def pin_memory(self, *args: List[str]):
        """
        Copies attributes to pinned memory, either for all attributes or
            only the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.pin_memory(), *args)

    def share_memory_(self, *args: List[str]):
        r"""Moves attributes to shared memory, either for all attributes or
        only the ones given in :obj:`*args`."""
        return self.apply_(lambda x: x.share_memory_(), *args)

    def detach_(self, *args: List[str]):
        """
        Detaches attributes from the computation graph, either for all
            attributes or only the ones given in :obj:`*args`.
        """
        return self.apply_(lambda x: x.detach_(), *args)

    def detach(self, *args: List[str]):
        """
        Detaches attributes from the computation graph by creating a new
            tensor, either for all attributes or only the ones given in
            :obj:`*args`.
        """
        return self.apply(lambda x: x.detach(), *args)

    def requires_grad_(self, *args: List[str], requires_grad: bool = True):
        """
        Tracks gradient computation, either for all attributes or only the
            ones given in :obj:`*args`.
        This function needs special treatment, as only Tensors of
            floating point dtype can require gradients.
        """
        return self.apply_(
            lambda x: (x.requires_grad_(requires_grad = requires_grad) if x.is_floating_point() else x),
            *args
        )

    def record_stream(self, stream: torch.cuda.Stream, *args: List[str]):
        """
        Ensures that the tensor memory is not reused for another tensor
            until all current work queued on :obj:`stream` has been completed,
            either for all attributes or only the ones given in :obj:`*args`.
        """
        for store in self.stores:
            store.record_stream(stream, *args)
        return self


    def is_cuda(self) -> bool:
        """
        Returns :obj:`True` if any :class:`torch.Tensor` or
            `torch_sparse.SparseTensor` attribute is stored
            on the GPU, :obj:`False` otherwise.
        """
        for store in self.stores:
            if store.is_cuda():
                return True
        return False

    def get_device(self):
        """
        Returns CUDA device id if any :class:`torch.Tensor` or
            `torch_sparse.SparseTensor` attribute is stored on the CUDA,
            cpu (-1) otherwise.
        """
        for store in self.stores:
            device = store.get_device()
            if device != -1:
                return device
        return -1

###############################################################################
@dataclass
class DataTensorAttr(TensorAttr):
    """Attribute class for `Data`, which does not require a `group_name`."""
    def __init__(
            self,
            attr_name = _field_status.UNSET,
            index = _field_status.UNSET,
    ):
        # Treat group_name as optional, and move it to the end
        super().__init__(None, attr_name, index)


@dataclass
class DataEdgeAttr(EdgeAttr):
    """
    Edge attribute class for `Data`, which does not require a edge_type`.
    """
    def __init__(
        self,
        layout: EdgeLayout,
        is_sorted: bool = False,
        size: Optional[Tuple[int, int]] = None,
        edge_type: EdgeType = None,
    ):
        # Treat edge_type as optional, and move it to the end
        super().__init__(edge_type, layout, is_sorted, size)


class Data(BaseData, FeatureStore, GraphStore):
    """
    A data object describing a homogeneous graph.
    The data object can hold node-level, link-level, graph-level and cv-level attributes.
    In general, :class:`Data` tries to mimic the behaviour of a regular Python dictionary.
    In addition, it provides useful functionality for analyzing graph
        structures, and provides basic PyTorch tensor functionalities.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
        introduction.html#data-handling-of-graphs>`__ for the accompanying
        tutorial.
    Args:
        x (Tensor, optional): Node feature matrix with shape :obj:`[num_nodes,
            num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph-level or node-level ground-truth labels
            with arbitrary shape. (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        img (Tensor, optional): Image matrix with shape :obj:`[Channnel, Height, Width]`.
            (default: :obj:`None`)
        **kwargs (optional): Additional attributes.
    """
    def __init__(
            self,
            x: OptInput = None,
            edge_index: OptInput = None,
            edge_attr: OptInput = None,
            y: OptInput = None,
            pos: OptInput = None,
            img: OptInput = None,
            **kwargs,
    ):
        # `Data` doesn't support group_name, so we need to adjust `TensorAttr`
        # accordingly here to avoid requiring `group_name` to be set:
        super().__init__(tensor_attr_cls = DataTensorAttr)

        # `Data` doesn't support edge_type, so we need to adjust `EdgeAttr`
        # accordingly here to avoid requiring `edge_type` to be set:
        GraphStore.__init__(self, edge_attr_cls = DataEdgeAttr)

        self.__dict__['_store'] = GlobalStorage(_parent = self)

        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y
        if pos is not None:
            self.pos = pos
        if img is not None:
            self.img = img

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        if '_store' not in self.__dict__:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, 'fset', None) is not None:
            propobj.fset(self, value)
        else:
            setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    # TODO consider supporting the feature store interface for
    # __getitem__, __setitem__, and __delitem__ so, for example, we
    # can accept key: Union[str, TensorAttr] in __getitem__.
    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any):
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __iter__(self) -> Iterable:
        """
        Iterates over all attributes in the data, yielding their attribute
            names and values.
        """
        for key, value in self._store.items():
            yield key, value

    def __call__(self, *args: List[str]) -> Iterable:
        """
        Iterates over all attributes :obj:`*args` in the data, yielding
            their attribute names and values.
        If :obj:`*args` is not given, will iterate over all attributes.
        """
        for key, value in self._store.items(*args):
            yield key, value

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_store'] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def __repr__(self) -> str:
        return _repr(self)

    ############################################## Data property

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    @property
    def cv_stores(self) -> List[CVStorage]:
        return [self._store]

    @property
    def meta_stores(self) -> List[Dict[str, Any]]:
        return [self._store.metastore]

    @property
    def x(self) -> Any:
        return self['x'] if 'x' in self._store else None

    @property
    def edge_index(self) -> Any:
        return self['edge_index'] if 'edge_index' in self._store else None

    @property
    def edge_weight(self) -> Any:
        return self['edge_weight'] if 'edge_weight' in self._store else None

    @property
    def edge_attr(self) -> Any:
        return self['edge_attr'] if 'edge_attr' in self._store else None

    @property
    def y(self) -> Any:
        return self['y'] if 'y' in self._store else None

    @property
    def pos(self) -> Any:
        return self['pos'] if 'pos' in self._store else None

    @property
    def img(self) -> Any:
        return self['img'] if 'img' in self._store else None

    @property
    def batch(self) -> Any:
        return self['batch'] if 'batch' in self._store else None

    @property
    def ptr(self) -> Any:
        return self['ptr'] if 'ptr' in self._store else None

    @property
    def num_node_features(self) -> int:
        """Returns the number of features per node in the graph."""
        return self._store.num_node_features

    @property
    def num_features(self) -> int:
        """Returns the number of features per node in the graph.
        Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        """Returns the number of features per edge in the graph."""
        return self._store.num_edge_features

    #######################################

    def __cat_dim__(self, key: str, value: DataContainer, *args, **kwargs) -> Any:
        # img data batchsize-wise stacked
        if value.stack:
            return None
        elif value.is_sptensor and 'adj' in key:
            return (0, 1)
        elif 'index' in key or key == 'face':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: DataContainer, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        else:
            return 0

    def validate(self, raise_on_error: bool = True) -> bool:
        """Validates the correctness of the data"""
        cls_name = self.__class__.__name__
        status = True

        num_nodes = self.num_nodes
        if num_nodes is None:
            status = False
            warn_or_raise(f"'num_nodes' is undefined in '{cls_name}'",
                          raise_on_error)

        if 'edge_index' in self and self.edge_index.numel() > 0:
            if self.edge_index.min < 0:
                status = False
                warn_or_raise(
                    f"'edge_index' contains negative indices in "
                    f"'{cls_name}' (found {int(self.edge_index.min)})",
                    raise_on_error)

            if num_nodes is not None and self.edge_index.max >= num_nodes:
                status = False
                warn_or_raise(
                    f"'edge_index' contains larger indices than the number "
                    f"of nodes ({num_nodes}) in '{cls_name}' "
                    f"(found {int(self.edge_index.max)})", raise_on_error)

        return status

    def is_node_attr(self, key: str) -> bool:
        """
        Returns :obj:`True` if the object at key :obj:`key` denotes a
            node-level attribute.
        """
        return self._store.is_node_attr(key)

    def is_edge_attr(self, key: str) -> bool:
        """
        Returns :obj:`True` if the object at key :obj:`key` denotes an
            edge-level attribute.
        """
        return self._store.is_edge_attr(key)

    def subgraph(self, subset: Union[Tensor, List[int]]):
        """
        Returns the induced subgraph given by the node indices :obj:`subset`.
        Args:
            subset (LongTensor or BoolTensor): The nodes to keep.
        """
        edge_index = self.edge_index.data
        if isinstance(subset, (list, tuple)):
            subset = torch.tensor(subset, dtype = torch.long, device = edge_index.device)

        out = subgraph(subset, edge_index, relabel_nodes=True,
                       num_nodes=self.num_nodes, return_edge_mask=True)
        edge_index, _, edge_mask = out

        if subset.dtype == torch.bool:
            num_nodes = int(subset.sum())
        else:
            num_nodes = subset.size(0)

        data = copy.copy(self)
        data.metastore['num_nodes'] = num_nodes
        for key, value in data:
            if key == 'edge_index':
                data.edge_index = self.edge_index.update(edge_index)
            elif value.is_tensor_or_sptensor:
                if self.is_node_attr(key):
                    data[key] = value.index(subset)
                elif self.is_edge_attr(key):
                    data[key] = value.index(edge_mask)

        return data

    def to_heterogeneous(
            self,
            node_type: Optional[Tensor] = None,
            edge_type: Optional[Tensor] = None,
            node_type_names: Optional[List[NodeType]] = None,
            edge_type_names: Optional[List[EdgeType]] = None,
    ):
        """
        Converts a :class:`~torch_geometric.data.Data` object to a
            heterogeneous :class:`~torch_geometric.data.HeteroData` object.
            For this, node and edge attributes are splitted according to the
            node-level and edge-level vectors :obj:`node_type` and
            :obj:`edge_type`, respectively.
        :obj:`node_type_names` and :obj:`edge_type_names` can be used to give
            meaningful node and edge type names, respectively.
        That is, the node_type :obj:`0` is given by :obj:`node_type_names[0]`.
        If the :class:`~torch_geometric.data.Data` object was constructed via
            :meth:`~torch_geometric.data.HeteroData.to_homogeneous`, the object can
            be reconstructed without any need to pass in additional arguments.
        Args:
            node_type (Tensor, optional): A node-level vector denoting the type
                of each node. (default: :obj:`None`)
            edge_type (Tensor, optional): An edge-level vector denoting the
                type of each edge. (default: :obj:`None`)
            node_type_names (List[str], optional): The names of node types.
                (default: :obj:`None`)
            edge_type_names (List[Tuple[str, str, str]], optional): The names
                of edge types. (default: :obj:`None`)
        **Note that this transform will lead to missing metastore.
        """
        from druglib.data import HeteroData
        # node_type, node type names, edge type and edge type names have
        # been moved into metastore
        if node_type is None:
            node_type: Optional[Tensor] = self._store.metastore.get('node_type', None)
        if node_type is None:
            node_type = torch.zeros(self.num_nodes, dtype=torch.long)

        if node_type_names is None:
            store = self._store
            node_type_names: List[NodeType] = store.metastore.get('_node_type_names', None)
        if node_type_names is None:
            node_type_names = [str(i) for i in node_type.unique().tolist()]

        if edge_type is None:
            edge_type: Optional[Tensor] = self._store.metastore.get('edge_type', None)
        if edge_type is None:
            edge_type = torch.zeros(self.num_edges, dtype=torch.long)

        if edge_type_names is None:
            store = self._store
            edge_type_names: List[EdgeType] = store.metastore.get('_edge_type_names', None)
        if edge_type_names is None:
            edge_type_names = []
            edge_index = self.edge_index.data
            for i in edge_type.unique().tolist():
                src, dst = edge_index[:, edge_type == i]
                src_types = node_type[src].unique().tolist()
                dst_types = node_type[dst].unique().tolist()
                if len(src_types) != 1 and len(dst_types) != 1:
                    raise ValueError(
                        "Could not construct a 'HeteroData' object from the "
                        "'Data' object because single edge types span over "
                        "multiple node types")
                edge_type_names.append((node_type_names[src_types[0]], str(i),
                                        node_type_names[dst_types[0]]))

        # We iterate over node types to find the local node indices belonging
        # to each node type. Furthermore, we create a global `index_map` vector
        # that maps global node indices to local ones in the final
        # heterogeneous graph:
        node_ids, index_map = {}, torch.empty_like(node_type)
        for i, key in enumerate(node_type_names):
            node_ids[i] = (node_type == i).nonzero(as_tuple=False).view(-1)
            index_map[node_ids[i]] = torch.arange(len(node_ids[i]),
                                                  device=index_map.device)

        # We iterate over edge types to find the local edge indices:
        edge_ids = {}
        for i, key in enumerate(edge_type_names):
            edge_ids[i] = (edge_type == i).nonzero(as_tuple=False).view(-1)

        data = HeteroData()

        for i, key in enumerate(node_type_names):
            for attr, value in self.items():
                if value.is_tensor and self.is_node_attr(attr):
                    data[key][attr] = value.index(node_ids[i])# DataContainer index function

            data[key].metastore['num_nodes'] = node_ids[i].size(0)

        for i, key in enumerate(edge_type_names):
            src, _, dst = key
            for attr, value in self.items():
                if attr == 'edge_index':
                    edge_index = value.data[:, edge_ids[i]]
                    edge_index[0] = index_map[edge_index[0]]
                    edge_index[1] = index_map[edge_index[1]]
                    data[key].edge_index = value.update(edge_index)
                elif value.is_tensor and self.is_edge_attr(attr):
                    data[key][attr] = value.index(edge_ids[i])

        # Add global attributes.
        keys = set(data.keys)
        for attr, value in self.items():
            if attr in keys:
                continue
            if attr in CVWORLD and any(k in attr for k in CVKEYS):
                data['cv'][attr] = value
                continue
            if len(data.node_stores) == 1:
                data.node_stores[0][attr] = value
            else:
                data[attr] = value

        return data

    ###########################################################################

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]):
        """
        Creates a :class:`~torch_geometric.data.Data` object from a Python dictionary.
        """
        return cls(**mapping)

    def to_dict(
            self,
            decode: bool = True,
            drop_meta: bool = False,
    ) -> Dict[str, Any]:
        return self._store.to_dict(decode, drop_meta)

    def to_namedtuple(self, decode: bool = True) -> NamedTuple:
        return self._store.to_namedtuple(decode)

    def stores_as(self, data: 'Data'):
        """Return self, actually this is placeholder for batching. Details see batch.py"""
        return self

    # FeatureStore interface ##################################################

    def items(self):
        """
        Returns an `ItemsView` over the stored attributes in the `Data` object.
        """
        # NOTE this is necessary to override the default `MutableMapping`
        # items() method.
        return self._store.items()

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        """Stores a feature tensor in node storage."""
        out = getattr(self, attr.attr_name, None)
        if out is not None and attr.index is not None:
            # Attr name exists, handle index:
            out[attr.index] = tensor.data if isinstance(tensor, DataContainer) else tensor
        else:
            # No attr name (or None index), just store tensor:
            # note that if tensor is :type:`Tensor` or :type:`SparseTensor`
            # the `setattr` mechanism of :object:`Data` can handle
            setattr(self, attr.attr_name, tensor)
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        """Obtains a feature tensor from node storage."""
        # Retrieve tensor and index accordingly:
        tensor = getattr(self, attr.attr_name, None)
        if tensor is not None:
            # TODO this behavior is a bit odd, since TensorAttr requires that
            # we set `index`. So, we assume here that indexing by `None` is
            # equivalent to not indexing at all, which is not in line with
            # Python semantics.
            return tensor[attr.index] if attr.index is not None else tensor
        return None

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        """Deletes a feature tensor from node storage."""
        # Remove tensor entirely:
        if hasattr(self, attr.attr_name):
            delattr(self, attr.attr_name)
            return True
        return False

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple:
        """Returns the size of the tensor corresponding to `attr`."""
        return self._get_tensor(attr).sizes()

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        """Obtains all feature attributes stored in `Data`."""
        return [
            TensorAttr(attr_name=name) for name in self._store.keys()
            if self._store.is_node_attr(name) or self._store.is_cv_attr(name)
        ]

    def __len__(self) -> int:
        return BaseData.__len__(self)

    # GraphStore interface ####################################################

    def _put_edge_index(
            self,
            edge_index: EdgeTensorType,
            edge_attr: EdgeAttr,
    ) -> bool:
        """
        Stores `edge_index` in `Data`, in the specified layout.
        Note that because input `edge_index` is a pair tuple if tensor,
            we do not enable to transform it to DataContainer, but keep
            it to a pair tuple of Tensor;
        Then, when the pair tuple of tensor transformed by `edge_tensor_type_to_adj_type`,
            actually it is a Tensor, so we transform it to DataContainer.
        """
        # Convert the edge index to a recognizable layout:
        attr_name = EDGE_LAYOUT_TO_ATTR_NAME[edge_attr.layout]
        attr_val = edge_tensor_type_to_adj_type(edge_attr, edge_index)
        attr_val = DataContainer(
            data = attr_val,
            stack = False,
            is_graph = True,
        )
        setattr(self, attr_name, attr_val)

        # Set edge attributes:
        if not hasattr(self, '_edge_attrs'):
            self._edge_attrs = {}

        self._edge_attrs[edge_attr.layout.value] = edge_attr

        # Set size, if possible:
        size = edge_attr.size
        if size is not None:
            if size[0] != size[1]:
                raise ValueError(
                    f"'Data' requires size[0] == size[1], but received "
                    f"the tuple {size}.")
            self.metastore['num_nodes'] = size[0]
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        """
        Obtains the edge index corresponding to `edge_attr` in `Data`,
            in the specified layout.
        """
        # Get the requested layout and the edge tensor type associated with it:
        attr_name = EDGE_LAYOUT_TO_ATTR_NAME[edge_attr.layout]
        attr_val = getattr(self._store, attr_name, None)
        if attr_val is not None:
            # Convert from Adj type DataContainer to Tuple[Tensor, Tensor]
            attr_val = adj_type_to_edge_tensor_type(edge_attr.layout, attr_val.data)
        return attr_val

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        """
        Returns `EdgeAttr` objects corresponding to the edge indices stored
            in `Data` and their layouts
        """
        if not hasattr(self, '_edge_attrs'):
            try:
                # Automatically create a `_edge_attrs` with coo format
                self._edge_attrs = {}
                self._edge_attrs['coo'] = EdgeAttr(
                    edge_type = None,
                    layout = EdgeLayout.COO,
                    size = (self.num_nodes, self.num_nodes)
                )
                setattr(self, 'coo', self.edge_index)
            except AttributeError:
                # generally, this is raised by call property `edge_index`
                # details see :cls:`EdgeStorage`.edge_index
                return []
        added_attrs = set()

        # Check edges added via _put_edge_index:
        edge_attrs = list(self._edge_attrs.values())
        for attr in edge_attrs:
            attr.size = (self.num_nodes, self.num_nodes)
            added_attrs.add(attr.layout)

        for layout, attr_name in EDGE_LAYOUT_TO_ATTR_NAME.items():
            if attr_name in self and layout not in added_attrs:
                edge_attrs.append(
                    EdgeAttr(
                        edge_type = None,
                        layout = layout,
                        size = (self.num_nodes, self.num_nodes)
                    )
                )

        return edge_attrs

###############################################################################

def _repr(value: Any, indent: int = 0) -> str:
    cls = value.__class__.__name__

    info = [size_repr(k, v.data) for k, v in value.items()]
    info = ', '.join(info)
    data_info = f'{cls}({info}),\n'

    meta_info = ''
    if len(value.metastore) > 0:
        info = [size_repr(k, v, indent = 2 + indent) for k, v in value.metastore.items()]
        meta_info = '\n' + ',\n'.join(info) + '\n' \
            if len(value.metastore) > 1 else f' {info[0]} '
    return f'({color.BOLD}{data_info}{color.END}' + ' metastore={' + f'{meta_info}' + ' })'

def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = ' ' * indent
    if isinstance(value, Tensor) and value.dim() == 0:
        out = value.item()
    elif isinstance(value, Tensor):
        out = str(list(value.size()))
    elif isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, SparseTensor):
        out = str(value.sizes())[:-1] + f', nnz={value.nnz()}]'
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, BaseStorage):
        out = _repr(value, indent)
    elif isinstance(value, Mapping) and len(value) == 0:
        out = '{}'
    elif (isinstance(value, Mapping) and len(value) == 1
          and not isinstance(list(value.values())[0], Mapping)):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = '{ ' + ', '.join(lines) + ' }'
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = '{\n' + ',\n'.join(lines) + '\n' + pad + '}'
    else:
        out = str(value)

    key = str(key).replace("'", '')
    if isinstance(value, BaseStorage):
        return f'{pad}{color.BOLD}{key}{color.END}={out}'
    else:
        return f'{pad}{key}={out}'


def warn_or_raise(msg: str, raise_on_error: bool = True):
    if raise_on_error:
        raise ValueError(msg)
    else:
        warnings.warn(msg)





