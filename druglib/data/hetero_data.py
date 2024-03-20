# Copyright (c) MDLDrugLib. All rights reserved.
import copy
import re
from collections import defaultdict, namedtuple
from collections.abc import Mapping
from itertools import chain
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor

from .data_container import DataContainer
from .data import BaseData, Data, size_repr, warn_or_raise
from .feature_store import FeatureStore, TensorAttr
from .graph_store import (
    EDGE_LAYOUT_TO_ATTR_NAME,
    EdgeAttr,
    GraphStore,
    adj_type_to_edge_tensor_type,
    edge_tensor_type_to_adj_type,
)
from .storage import BaseStorage, EdgeStorage, NodeStorage, CVStorage
from .typing import (
    EdgeTensorType,
    EdgeType,
    FeatureTensorType,
    NodeType,
    QueryType,
    PairTensor,
    CVWORLD,
    CVKEYS,
)
from torch_geometric.utils import is_undirected, index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes

NodeOrEdgeType = Union[NodeType, EdgeType]
NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]


class HeteroData(BaseData, FeatureStore, GraphStore):
    """
    A data object describing a heterogeneous graph, holding multiple node
        and/or edge types in disjunct storage objects.
    Storage objects can hold either node-level, link-level or graph-level
        attributes.
    In general, :class:`HeteroData` tries to mimic the behaviour of
        a regular **nested** Python dictionary.
    In addition, it provides useful functionality for analyzing graph
        structures, and provides basic PyTorch tensor functionalities.
    .. code-block::
        data = HeteroData()
        # Create two node types "paper" and "author" holding a feature matrix:
        data['paper'].x = torch.randn(num_papers, num_paper_features)
        data['author'].x = torch.randn(num_authors, num_authors_features)
        # Create an edge type "(author, writes, paper)" and building the
        # graph connectivity:
        data['author', 'writes', 'paper'].edge_index = ...  # [2, num_edges]
        data['paper'].num_nodes
        >>> 23
        data['author', 'writes', 'paper'].num_edges
        >>> 52
        # PyTorch tensor functionality:
        data = data.pin_memory()
        data = data.to('cuda:0', non_blocking=True)
    Note that there exists multiple ways to create a heterogeneous graph data,
    *e.g.*:
    * To initialize a node of type :obj:`"paper"` holding a node feature
      matrix :obj:`x_paper` named :obj:`x`:
      .. code-block:: python
        data = HeteroData()
        data['paper'].x = x_paper
        data = HeteroData(paper={ 'x': x_paper })
        data = HeteroData({'paper': { 'x': x_paper }})
    * To initialize an edge from source node type :obj:`"author"` to
      destination node type :obj:`"paper"` with relation type :obj:`"writes"`
      holding a graph connectivity matrix :obj:`edge_index_author_paper` named
      :obj:`edge_index`:
      .. code-block:: python
        data = HeteroData()
        data['author', 'writes', 'paper'].edge_index = edge_index_author_paper
        data = HeteroData(author__writes__paper={
            'edge_index': edge_index_author_paper
        })
        data = HeteroData({
            ('author', 'writes', 'paper'):
            { 'edge_index': edge_index_author_paper }
        })
    """

    DEFAULT_REL = 'to'

    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__()

        self.__dict__['_global_store'] = BaseStorage(_parent = self)
        self.__dict__['_node_store_dict'] = {}
        self.__dict__['_edge_store_dict'] = {}
        self.__dict__['_cv_store_dict'] = {}# though it is dict, only support cv key

        for key, value in chain((_mapping or {}).items(), kwargs.items()):
            if '__' in key and isinstance(value, Mapping):
                key = tuple(key.split('__'))

            if isinstance(value, Mapping):
                self[key].update(value)
            else:
                setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        # `data.*_dict` => Link to node and edge stores.
        # `data.*` => Link to the `_global_store`.
        # Using `data.*_dict` is the same as using `collect()` for collecting
        # nodes and edges features.
        if hasattr(self._global_store, key):
            return getattr(self._global_store, key)
        elif bool(re.search('_dict$', key)):
            return self.collect(key[:-5])
        raise AttributeError(f"'{self.__class__.__name__}' has no "
                             f"attribute '{key}'")

    def __setattr__(self, key: str, value: FeatureTensorType):
        # NOTE: We aim to prevent duplicates in node, edge or cv types.
        if key in self.cv_types:
            assert AttributeError(f"'{key}' is already present as a cv type")
        elif key in self.node_types:
            raise AttributeError(f"'{key}' is already present as a node type")
        elif key in self.edge_types:
            raise AttributeError(f"'{key}' is already present as an edge type")
        setattr(self._global_store, key, value)

    def __delattr__(self, key: str):
        delattr(self._global_store, key)

    def __getitem__(self, *args: Tuple[QueryType]) -> Any:
        # `data[*]` => Link to either `_global_store`, _node_store_dict` or
        # `_edge_store_dict`.
        # If neither is present, we create a new `Storage` object for the given
        # node/edge-type.
        key = self._to_canonical(*args)

        out = self._global_store.get(key, None)
        if out is not None:
            return out

        if isinstance(key, tuple):
            return self.get_edge_store(*key)
        else:
            if key == 'cv':
                return self.get_cv_store(key)
            return self.get_node_store(key)

    def __setitem__(self, key: str, value: FeatureTensorType):
        if key in self.cv_types:
            assert AttributeError(f"'{key}' is already present as a cv type")
        elif key in self.node_types:
            raise AttributeError(f"'{key}' is already present as a node type")
        elif key in self.edge_types:
            raise AttributeError(f"'{key}' is already present as an edge type")
        self._global_store[key] = value

    def __delitem__(self, *args: Tuple[QueryType]):
        # `del data[*]` => Link to `_node_store_dict` or `_edge_store_dict`.
        key = self._to_canonical(*args)
        if key in self.edge_types:
            del self._edge_store_dict[key]
        elif key in self.node_types:
            del self._node_store_dict[key]
        elif key in self.cv_types:
            del self._cv_store_dict[key]
        else:
            del self._global_store[key]

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_global_store'] = copy.copy(self._global_store)
        out._global_store._parent = out
        out.__dict__['_node_store_dict'] = {}
        for key, store in self._node_store_dict.items():
            out._node_store_dict[key] = copy.copy(store)
            out._node_store_dict[key]._parent = out
        out.__dict__['_edge_store_dict'] = {}
        for key, store in self._edge_store_dict.items():
            out._edge_store_dict[key] = copy.copy(store)
            out._edge_store_dict[key]._parent = out
        out.__dict__['_cv_store_dict'] = {}
        for key, store in self._cv_store_dict.items():
            out._cv_store_dict[key] = copy.copy(store)
            out._cv_store_dict[key]._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._global_store._parent = out
        for key in self._node_store_dict.keys():
            out._node_store_dict[key]._parent = out
        for key in out._edge_store_dict.keys():
            out._edge_store_dict[key]._parent = out
        for key in out._cv_store_dict.keys():
            out._cv_store_dict[key]._parent = out
        return out

    def __repr__(self) -> str:
        info1 = [size_repr(k, v, 2) for k, v in self._global_store.items()]
        info2 = [size_repr(k, v, 2) for k, v in self._node_store_dict.items()]
        info3 = [size_repr(k, v, 2) for k, v in self._edge_store_dict.items()]
        info4 = [size_repr(k, v, 2) for k, v in self._cv_store_dict.items()]
        info = ',\n'.join(info1 + info2 + info3 + info4)
        info = f'\n{info}\n' if len(info) > 0 else info
        return f'{self.__class__.__name__}({info})'

    def stores_as(self, data: 'HeteroData'):
        for node_type in data.node_types:
            self.get_node_store(node_type)
        for edge_type in data.edge_types:
            self.get_edge_store(*edge_type)
        for cv_type in data.cv_types:
            self.get_cv_store(cv_type)
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        """Returns a list of all storages of the graph."""
        return ([self._global_store] + list(self.node_stores) +
                list(self.edge_stores) + list(self.cv_stores))

    @property
    def node_types(self) -> List[NodeType]:
        """Returns a list of all node types of the graph."""
        return list(self._node_store_dict.keys())

    @property
    def node_stores(self) -> List[NodeStorage]:
        """Returns a list of all node storages of the graph."""
        return list(self._node_store_dict.values())

    @property
    def edge_types(self) -> List[EdgeType]:
        """Returns a list of all edge types of the graph."""
        return list(self._edge_store_dict.keys())

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        """Returns a list of all edge storages of the graph."""
        return list(self._edge_store_dict.values())

    @property
    def cv_types(self) -> List[str]:
        """Returns a sing-element "cv" or None list"""
        keys = list(self._cv_store_dict.keys())
        return keys

    @property
    def cv_stores(self) -> List[CVStorage]:
        """Returns a list of all cv storages of HeterData"""
        return list(self._cv_store_dict.values())

    def node_items(self) -> List[Tuple[NodeType, NodeStorage]]:
        """Returns a list of node type and node storage pairs."""
        return list(self._node_store_dict.items())

    def edge_items(self) -> List[Tuple[EdgeType, EdgeStorage]]:
        """Returns a list of edge type and edge storage pairs."""
        return list(self._edge_store_dict.items())

    def cv_items(self) -> List[Tuple[str, CVStorage]]:
        """Returns a list of "cv" and cv storage pairs."""
        return list(self._cv_store_dict.items())

    def to_dict(self) -> Dict[str, Any]:
        out = self._global_store.to_dict()
        for key, store in chain(self._node_store_dict.items(),
                                self._edge_store_dict.items(),
                                self._cv_store_dict.items()):
            out[key] = store.to_dict()
        return out

    def to_namedtuple(self) -> NamedTuple:
        field_names = list(self._global_store.keys())
        field_values = list(self._global_store.values())
        field_names += [
            '__'.join(key) if isinstance(key, tuple) else key
            for key in self.node_types + self.edge_types + self.cv_types
        ]
        field_values += [
            store.to_namedtuple()
            for store in self.node_stores + self.edge_stores + self.cv_stores
        ]
        DataTuple = namedtuple('DataTuple', field_names)
        return DataTuple(*field_values)

    def __cat_dim__(self, key: str, value: DataContainer,
                    store: Optional[NodeOrEdgeStorage] = None,
                    *args, **kwargs) -> Any:
        if value.stack:
            return None
        elif value.is_sptensor and 'adj' in key:
            return (0, 1)
        elif 'index' in key or key == 'face':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any,
                store: Optional[NodeOrEdgeStorage] = None,
                *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif isinstance(store, EdgeStorage) and 'index' in key:
            return torch.tensor(store.size()).view(2, 1)
        else:
            return 0

    @property
    def num_nodes(self) -> Optional[int]:
        """Returns the number of nodes in the graph."""
        return super().num_nodes

    @property
    def num_node_features(self) -> Dict[NodeType, int]:
        """Returns the number of features per node type in the graph."""
        return {
            key: store.num_node_features
            for key, store in self._node_store_dict.items()
        }

    @property
    def num_features(self) -> Dict[NodeType, int]:
        """
        Returns the number of features per node type in the graph.
        Alias for :py:attr:`~num_node_features`.
        """
        return self.num_node_features

    @property
    def num_edge_features(self) -> Dict[EdgeType, int]:
        """Returns the number of features per edge type in the graph."""
        return {
            key: store.num_edge_features
            for key, store in self._edge_store_dict.items()
        }

    @property
    def cvchannels(self) -> Dict[EdgeType, int]:
        """Returns the number of img data channel"""
        return {
            key: store.channels
            for key, store in self._cv_store_dict.items()
        }

    @property
    def cvsize(self) -> Dict[EdgeType, Union[torch.Size, int, None]]:
        """Returns the number of img data channel"""
        return {
            key: store.cvsize()
            for key, store in self._cv_store_dict.items()
        }

    def is_undirected(self) -> bool:
        """Returns :obj:`True` if graph edges are undirected."""
        edge_index, _, _ = to_homogeneous_edge_index(self)
        # edge index is tensor rather than DataContainer
        return is_undirected(edge_index, num_nodes=self.num_nodes)

    def validate(self, raise_on_error: bool = True) -> bool:
        """Validates the correctness of the data."""
        cls_name = self.__class__.__name__
        status = True

        for edge_type, store in self._edge_store_dict.items():
            src, _, dst = edge_type

            num_src_nodes = self[src].num_nodes
            num_dst_nodes = self[dst].num_nodes
            if num_src_nodes is None:
                status = False
                warn_or_raise(
                    f"'num_nodes' is undefined in node type '{src}' of "
                    f"'{cls_name}'", raise_on_error)

            if num_dst_nodes is None:
                status = False
                warn_or_raise(
                    f"'num_nodes' is undefined in node type '{dst}' of "
                    f"'{cls_name}'", raise_on_error)

            if 'edge_index' in store and store.edge_index.numel() > 0:
                if store.edge_index.min() < 0:
                    status = False
                    warn_or_raise(
                        f"'edge_index' of edge type {edge_type} contains "
                        f"negative indices in '{cls_name}' "
                        f"(found {int(store.edge_index.min())})",
                        raise_on_error)

                if (num_src_nodes is not None
                        and store.edge_index[0].max() >= num_src_nodes):
                    status = False
                    warn_or_raise(
                        f"'edge_index' of edge type {edge_type} contains"
                        f"larger source indices than the number of nodes"
                        f"({num_src_nodes}) of this node type in '{cls_name}' "
                        f"(found {int(store.edge_index[0].max())})",
                        raise_on_error)

                if (num_dst_nodes is not None
                        and store.edge_index[1].max() >= num_dst_nodes):
                    status = False
                    warn_or_raise(
                        f"'edge_index' of edge type {edge_type} contains"
                        f"larger destination indices than the number of nodes"
                        f"({num_dst_nodes}) of this node type in '{cls_name}' "
                        f"(found {int(store.edge_index[1].max())})",
                        raise_on_error)

        return status

    ###########################################################################

    def _to_canonical(self, *args: Tuple[QueryType]) -> NodeOrEdgeType:
        # Converts a given `QueryType` to its "canonical type":
        # 1. `relation_type` will get mapped to the unique
        #    `(src_node_type, relation_type, dst_node_type)` tuple.
        # 2. `(src_node_type, dst_node_type)` will get mapped to the unique
        #    `(src_node_type, *, dst_node_type)` tuple, and
        #    `(src_node_type, 'to', dst_node_type)` otherwise.
        if len(args) == 1:
            args = args[0]

        if isinstance(args, str):
            # a cv type
            if args == 'cv':
                return args

            # a node key
            node_types = [key for key in self.node_types if key == args]
            if len(node_types) == 1:
                args = node_types[0]
                return args

            # a edge relation type
            # Try to map to edge type based on unique relation type:
            edge_types = [key for key in self.edge_types if key[1] == args]
            if len(edge_types) == 1:
                args = edge_types[0]
                return args

        elif len(args) == 2:
            # a pair tuple edge type
            # Try to find the unique source/destination node tuple:
            edge_types = [
                key for key in self.edge_types
                if key[0] == args[0] and key[-1] == args[-1]
            ]
            if len(edge_types) == 1:
                args = edge_types[0]
                return args
            elif len(edge_types) == 0:
                args = (args[0], self.DEFAULT_REL, args[1])
                return args
        # unknown node type and edge type return
        return args

    def metadata(self) -> Tuple[List[NodeType], List[EdgeType], List[str]]:
        """
        Returns the heterogeneous meta-data, *i.e.* its node, edge and cv types.
        .. code-block:: python
            data = HeteroData()
            data['paper'].x = ...
            data['author'].x = ...
            data['author', 'writes', 'paper'].edge_index = ...
            data['cv'].img = ...
            print(data.metadata())
            >>> (['paper', 'author'], [('author', 'writes', 'paper')], ['cv'])
        """
        return self.node_types, self.edge_types, self.cv_types

    def collect(self, key: str) -> Dict[NodeOrEdgeType, Any]:
        """
        Collects the attribute :attr:`key` from all node, edge and cv types.
        .. code-block:: python
            data = HeteroData()
            data['paper'].x = ...
            data['author'].x = ...
            print(data.collect('x'))
            >>> { 'paper': ..., 'author': ...}
        .. note::
            This is equivalent to writing :obj:`data.x_dict`.
        """
        mapping = {}
        for subtype, store in chain(self._node_store_dict.items(),
                                    self._edge_store_dict.items(),
                                    self._cv_store_dict.items()):
            if hasattr(store, key):
                mapping[subtype] = getattr(store, key)
        return mapping

    def get_node_store(self, key: NodeType) -> NodeStorage:
        """
        Gets the :class:`NodeStorage` object of a particular node type :attr:`key`.
        If the storage is not present yet, will create a new :class:`NodeStorage`
            object for the given node type.
        .. code-block:: python
            data = HeteroData()
            node_storage = data.get_node_store('paper')
        """
        # impede 'cv' type in
        assert key != 'cv', 'while node type is needed, cv type is input.'
        out = self._node_store_dict.get(key, None)
        if out is None:
            out = NodeStorage(_parent = self, _key = key)
            self._node_store_dict[key] = out
        return out

    def get_edge_store(self, src: str, rel: str, dst: str) -> EdgeStorage:
        """
        Gets the :class:`EdgeStorage` object of a particular edge type
            given by the tuple :obj:`(src, rel, dst)`.
        If the storage is not present yet, will create a new :class:`EdgeStorage`
            object for the given edge type.
        .. code-block:: python
            data = HeteroData()
            edge_storage = data.get_edge_store('author', 'writes', 'paper')
        """
        key = (src, rel, dst)
        out = self._edge_store_dict.get(key, None)
        if out is None:
            out = EdgeStorage(_parent = self, _key = key)
            self._edge_store_dict[key] = out
        return out

    def get_cv_store(self, key: str) -> CVStorage:
        """
        Gets the :class:`CVStorage` object by given the string key 'cv'.
        If the storage is not present yet, will create a new :class:`CVStorage`
            object for the given cv type.
        .. code-block:: python
            data = HeteroData()
            cv_storage = data.get_cv_store('cv')
        """
        # Ensure key is 'cv', this allow a future extension
        assert key == 'cv', "Current cv store only supports 'cv' type."
        out = self._cv_store_dict.get(key, None)
        if out is None:
            out = CVStorage(_parent=self, _key = key)
            self._cv_store_dict[key] = out
        return out

    def rename(self, name: NodeType, new_name: NodeType) -> 'HeteroData':
        """
        Renames the node type :obj:`name` to :obj:`new_name` in-place.
        **Note that it exclusively belongs to graph data.
        """
        node_store = self._node_store_dict.pop(name)
        node_store._key = new_name
        self._node_store_dict[new_name] = node_store

        for edge_type in self.edge_types:
            src, rel, dst = edge_type
            if src == name or dst == name:
                edge_store = self._edge_store_dict.pop(edge_type)
                src = new_name if src == name else src
                dst = new_name if dst == name else dst
                edge_type = (src, rel, dst)
                edge_store._key = edge_type
                self._edge_store_dict[edge_type] = edge_store

        return self

    def subgraph(self, subset_dict: Dict[NodeType, Tensor]) -> 'HeteroData':
        """
        Returns the induced subgraph containing the node types and
            corresponding nodes in :obj:`subset_dict`.
        .. code-block:: python
            data = HeteroData()
            data['paper'].x = ...
            data['author'].x = ...
            data['conference'].x = ...
            data['paper', 'cites', 'paper'].edge_index = ...
            data['author', 'paper'].edge_index = ...
            data['paper', 'conference'].edge_index = ...
            print(data)
            >>> HeteroData(
                paper={ x=[10, 16] },
                author={ x=[5, 32] },
                conference={ x=[5, 8] },
                (paper, cites, paper)={ edge_index=[2, 50] },
                (author, to, paper)={ edge_index=[2, 30] },
                (paper, to, conference)={ edge_index=[2, 25] }
            )
            subset_dict = {
                'paper': torch.tensor([3, 4, 5, 6]),
                'author': torch.tensor([0, 2]),
            }
            print(data.subgraph(subset_dict))
            >>> HeteroData(
                paper={ x=[4, 16] },
                author={ x=[2, 32] },
                (paper, cites, paper)={ edge_index=[2, 24] },
                (author, to, paper)={ edge_index=[2, 5] }
            )
        Args:
            subset_dict (Dict[str, LongTensor or BoolTensor]): A dictonary
                holding the nodes to keep for each node type.
        **Note that it exclusively belongs to graph data.
        """
        data = self.__class__(self._global_store)

        for node_type, subset in subset_dict.items():
            if subset.dtype == torch.bool:
                data[node_type].metastore['num_nodes'] = int(subset.sum())
            else:
                data[node_type].metastore['num_nodes'] = subset.size(0)
            for key, value in self[node_type].items():
                if self[node_type].is_node_attr(key):
                    data[node_type][key] = value.index(subset)
                else:
                    data[node_type][key] = value

        for edge_type in self.edge_types:
            src, _, dst = edge_type
            if src not in subset_dict or dst not in subset_dict:
                continue

            edge_index, _, edge_mask = bipartite_subgraph(
                (subset_dict[src], subset_dict[dst]),
                self[edge_type].edge_index.data,
                relabel_nodes = True,
                size = (self[src].num_nodes, self[dst].num_nodes),
                return_edge_mask = True,
            )

            for key, value in self[edge_type].items():
                if key == 'edge_index':
                    data[edge_type].edge_index = self[edge_type].update(edge_index)
                elif self[edge_type].is_edge_attr(key):
                    data[edge_type][key] = value.index(edge_mask)
                else:
                    data[edge_type][key] = value

        return data

    def node_type_subgraph(self, node_types: List[NodeType]) -> 'HeteroData':
        """
        Returns the subgraph induced by the given :obj:`node_types`, *i.e.*
            the returned :class:`HeteroData` object only contains the node types
            which are included in :obj:`node_types`, and only contains the edge
            types where both end points are included in :obj:`node_types`.
        """
        data = copy.copy(self)
        for edge_type in self.edge_types:
            src, _, dst = edge_type
            if src not in node_types or dst not in node_types:
                del data[edge_type]
        for node_type in self.node_types:
            if node_type not in node_types:
                del data[node_type]
        return data

    def edge_type_subgraph(self, edge_types: List[EdgeType]) -> 'HeteroData':
        """
        Returns the subgraph induced by the given :obj:`edge_types`, *i.e.*
            the returned :class:`HeteroData` object only contains the edge types
            which are included in :obj:`edge_types`, and only contains the node
            types of the end points which are included in :obj:`node_types`.
        """
        edge_types = [self._to_canonical(e) for e in edge_types]

        data = copy.copy(self)
        for edge_type in self.edge_types:
            if edge_type not in edge_types:
                del data[edge_type]
        node_types = set(e[0] for e in edge_types)
        node_types |= set(e[-1] for e in edge_types)
        for node_type in self.node_types:
            if node_type not in node_types:
                del data[node_type]
        return data

    def to_homogeneous(self, node_attrs: Optional[List[str]] = None,
                       edge_attrs: Optional[List[str]] = None,
                       cv_attrs: Optional[List[str]] = None,
                       add_node_type: bool = True,
                       add_edge_type: bool = True) -> Data:
        """
        Converts a :class:`HeteroData` object to a homogeneous :class:`Data` object.
        By default, all features with same feature dimensionality across
            different types will be merged into a single representation, unless
            otherwise specified via the :obj:`node_attrs` and :obj:`edge_attrs`
            arguments.
        Furthermore, attributes named :obj:`node_type` and :obj:`edge_type`
            will be added to the returned :class:`Data` object, denoting
            node-level and edge-level vectors holding the node and edge
            type as integers, respectively.
        Args:
            node_attrs (List[str], optional): The node features to combine
                across all node types. These node features need to be of the
                same feature dimensionality. If set to :obj:`None`, will
                automatically determine which node features to combine.
                (default: :obj:`None`)
            edge_attrs (List[str], optional): The edge features to combine
                across all edge types. These edge features need to be of the
                same feature dimensionality. If set to :obj:`None`, will
                automatically determine which edge features to combine.
                (default: :obj:`None`)
            cv_attrs (List[str], optional): The cv img features to combine
                across the 'cv' type. These cv features do not need to be of
                the same feature dimensionality because no concatenation is needed.
                If set to :obj:`None`, will automatically determine which cv features
                to combine. (default: :obj:`None`)
            add_node_type (bool, optional): If set to :obj:`False`, will not
                add the node-level vector :obj:`node_type` to the returned
                :class:`~torch_geometric.data.Data` object.
                (default: :obj:`True`)
            add_edge_type (bool, optional): If set to :obj:`False`, will not
                add the edge-level vector :obj:`edge_type` to the returned
                :class:`~torch_geometric.data.Data` object.
                (default: :obj:`True`)
        **Note that this transform will lead to missing metastore for every type.
        """
        def _consistent_size(stores: List[BaseStorage]) -> List[str]:
            sizes_dict = defaultdict(list)
            for store in stores:
                for key, value in store.items():
                    if key in ['edge_index', 'adj_t', 'adj']:
                        continue
                    if value.is_tensor:
                        dim = self.__cat_dim__(key, value, store)
                        size = value.sizes()[:dim] + value.sizes()[dim + 1:]
                        sizes_dict[key].append(tuple(size))
            return [
                k for k, sizes in sizes_dict.items()
                if len(sizes) == len(stores) and len(set(sizes)) == 1
            ]

        edge_index, node_slices, edge_slices = to_homogeneous_edge_index(self)
        device = edge_index.device if edge_index is not None else None

        data = Data(**self._global_store.to_dict())
        if edge_index is not None:
            data.edge_index = edge_index
        data.metastore['_node_type_names'] = list(node_slices.keys())
        data.metastore['_edge_type_names'] = list(edge_slices.keys())

        # Combine node attributes into a single tensor:
        if node_attrs is None:
            node_attrs = _consistent_size(self.node_stores)
        for key in node_attrs:
            dcs = [store[key] for store in self.node_stores]
            values = [dc.data for dc in dcs]
            dim = self.__cat_dim__(key, dcs[0], self.node_stores[0])
            value = dcs[0].update(torch.cat(values, dim)) if len(values) > 1 else dcs[0]
            data[key] = value

        if not data.can_infer_num_nodes:
            data.metastore['num_nodes'] = list(node_slices.values())[-1][1]

        # Combine edge attributes into a single tensor:
        if edge_attrs is None:
            edge_attrs = _consistent_size(self.edge_stores)
        for key in edge_attrs:
            dcs = [store[key] for store in self.edge_stores]
            values = [dc.data for dc in dcs]
            dim = self.__cat_dim__(key, values[0], self.edge_stores[0])
            value = dcs[0].update(torch.cat(values, dim)) if len(values) > 1 else dcs[0]
            data[key] = value

        # Combine cv type attributes, a single type, no concatenation is needed
        if cv_attrs is None:
            cv_attrs = []
            for store in self.cv_stores:
                for k in store.keys():
                    cv_attrs.append(k)
        for key in cv_attrs:
            assert key in CVWORLD and any(k in key for k in CVKEYS)
            # avoid cv_attrs is not None, but cv_stores is None.
            if len(self.cv_stores) == 0:
                continue
            data[key] = self.cv_stores[0][key]


        if add_node_type:
            sizes = [offset[1] - offset[0] for offset in node_slices.values()]
            sizes = torch.tensor(sizes, dtype=torch.long, device=device)
            node_type = torch.arange(len(sizes), device=device)
            data.metastore['node_type'] = node_type.repeat_interleave(sizes)

        if add_edge_type and edge_index is not None:
            sizes = [offset[1] - offset[0] for offset in edge_slices.values()]
            sizes = torch.tensor(sizes, dtype=torch.long, device=device)
            edge_type = torch.arange(len(sizes), device=device)
            data.metastore['edge_type'] = edge_type.repeat_interleave(sizes)

        return data

    # FeatureStore interface ##################################################

    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        """Stores a feature tensor in node storage."""
        if not attr.is_set('index'):
            attr.index = None

        out = self._node_store_dict.get(attr.group_name, None)
        if out:
            # Group name exists, handle index or create new attribute name:
            val = getattr(out, attr.attr_name, None)
            if val is not None:
                val[attr.index] = tensor.data if isinstance(tensor, DataContainer) else tensor
            else:
                setattr(self[attr.group_name], attr.attr_name, tensor)
        else:
            # No node storage found, just store tensor in new one:
            setattr(self[attr.group_name], attr.attr_name, tensor)
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        """Obtains a feature tensor from node storage."""
        # Retrieve tensor and index accordingly:
        tensor = getattr(self[attr.group_name], attr.attr_name, None)
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
        if hasattr(self[attr.group_name], attr.attr_name):
            delattr(self[attr.group_name], attr.attr_name)
            return True
        return False

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple:
        """Returns the size of the tensor corresponding to `attr`."""
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        out = []
        for group_name, group in chain(self.node_items(),
                                       self.cv_items()):
            for attr_name in group:
                if group.is_node_attr(attr_name) or group.is_cv_attr(attr_name):
                    out.append(TensorAttr(group_name, attr_name))
        return out

    def __len__(self) -> int:
        return BaseData.__len__(self)

    def __iter__(self):
        raise NotImplementedError

    # GraphStore interface ####################################################

    def _put_edge_index(self, edge_index: EdgeTensorType,
                        edge_attr: EdgeAttr) -> bool:
        """Stores an edge index in edge storage, in the specified layout."""

        # Convert the edge index to a recognizable layout:
        attr_name = EDGE_LAYOUT_TO_ATTR_NAME[edge_attr.layout]
        attr_val = edge_tensor_type_to_adj_type(edge_attr, edge_index)
        attr_val = DataContainer(
            data = attr_val,
            stack = False,
            is_graph = True,
        )
        setattr(self[edge_attr.edge_type], attr_name, attr_val)

        # Set edge attributes:
        if not hasattr(self[edge_attr.edge_type], '_edge_attrs'):
            self[edge_attr.edge_type]._edge_attrs = {}

        self[edge_attr.edge_type]._edge_attrs[
            edge_attr.layout.value] = edge_attr

        key = self._to_canonical(edge_attr.edge_type)
        src, _, dst = key

        # Handle num_nodes, if possible:
        size = edge_attr.size
        if size is not None:
            # TODO better warning in the case of overwriting 'num_nodes'
            self[src].metastore['num_nodes'] = size[0]
            self[dst].metastore['num_nodes'] = size[1]

        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        """Gets an edge index from edge storage, in the specified layout."""
        # Get the requested layout and the Adj tensor associated with it:
        attr_name = EDGE_LAYOUT_TO_ATTR_NAME[edge_attr.layout]
        attr_val = getattr(self[edge_attr.edge_type], attr_name, None)
        if attr_val is not None:
            # Convert from Adj type to Tuple[Tensor, Tensor]
            attr_val = adj_type_to_edge_tensor_type(edge_attr.layout, attr_val.data)
        return attr_val

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        """
        Returns a list of `EdgeAttr` objects corresponding to the edge
            indices stored in `HeteroData` and their layouts.
        """
        out = []
        added_attrs = set()

        # Check edges added via _put_edge_index:
        for edge_type, _ in self.edge_items():
            if not hasattr(self[edge_type], '_edge_attrs'):
                continue
            edge_attrs = self[edge_type]._edge_attrs.values()
            for attr in edge_attrs:
                attr.size = self[edge_type].size()
                added_attrs.add((attr.edge_type, attr.layout))
            out.extend(edge_attrs)

        # Check edges added through regular interface:
        # TODO deprecate this and store edge attributes for all edges in
        # EdgeStorage
        for edge_type, edge_store in self.edge_items():
            for layout, attr_name in EDGE_LAYOUT_TO_ATTR_NAME.items():
                # Don't double count:
                if attr_name in edge_store and ((edge_type, layout)
                                                not in added_attrs):
                    out.append(
                        EdgeAttr(edge_type=edge_type, layout=layout,
                                 size=self[edge_type].size()))

        return out


# Helper functions ############################################################


def to_homogeneous_edge_index(
    data: HeteroData,
) -> Tuple[Optional[Tensor], Dict[NodeType, Any], Dict[EdgeType, Any]]:
    # Record slice information per node type:
    cumsum = 0
    node_slices: Dict[NodeType, Tuple[int, int]] = {}
    for node_type, store in data._node_store_dict.items():
        num_nodes = store.num_nodes
        node_slices[node_type] = (cumsum, cumsum + num_nodes)
        cumsum += num_nodes

    # Record edge indices and slice information per edge type:
    cumsum = 0
    edge_indices: List[Tensor] = []
    edge_slices: Dict[EdgeType, Tuple[int, int]] = {}
    for edge_type, store in data._edge_store_dict.items():
        src, _, dst = edge_type
        edge_index = store.edge_index.data
        offset = [[node_slices[src][0]], [node_slices[dst][0]]]
        offset = torch.tensor(offset, device = edge_index.device)
        edge_indices.append(edge_index + offset)

        num_edges = store.num_edges
        edge_slices[edge_type] = (cumsum, cumsum + num_edges)
        cumsum += num_edges

    edge_index = None
    if len(edge_indices) == 1:  # Memory-efficient `torch.cat`:
        edge_index = edge_indices[0]
    elif len(edge_indices) > 0:
        edge_index = torch.cat(edge_indices, dim=-1)

    return edge_index, node_slices, edge_slices


def bipartite_subgraph(
    subset: Union[PairTensor, Tuple[List[int], List[int]]],
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    relabel_nodes: bool = False,
    size: Tuple[int, int] = None,
    return_edge_mask: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Returns the induced subgraph of the bipartite graph
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.
    Args:
        subset (Tuple[Tensor, Tensor] or tuple([int],[int])): The nodes
            to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        size (tuple, optional): The number of nodes.
            (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    device = edge_index.device

    if isinstance(subset[0], (list, tuple)):
        subset = (torch.tensor(subset[0], dtype=torch.long, device=device),
                  torch.tensor(subset[1], dtype=torch.long, device=device))

    if subset[0].dtype == torch.bool or subset[0].dtype == torch.uint8:
        size = subset[0].size(0), subset[1].size(0)
    else:
        if size is None:
            size = (maybe_num_nodes(edge_index[0]),
                    maybe_num_nodes(edge_index[1]))
        subset = (index_to_mask(subset[0], size=size[0]),
                  index_to_mask(subset[1], size=size[1]))

    node_mask = subset
    edge_mask = node_mask[0][edge_index[0]] & node_mask[1][edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx_i = torch.zeros(node_mask[0].size(0), dtype=torch.long,
                                 device=device)
        node_idx_j = torch.zeros(node_mask[1].size(0), dtype=torch.long,
                                 device=device)
        node_idx_i[node_mask[0]] = torch.arange(node_mask[0].sum().item(),
                                                device=device)
        node_idx_j[node_mask[1]] = torch.arange(node_mask[1].sum().item(),
                                                device=device)
        edge_index = torch.stack(
            [node_idx_i[edge_index[0]], node_idx_j[edge_index[1]]])

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr
