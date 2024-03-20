# Copyright (c) MDLDrugLib. All rights reserved.
# Modified from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/storage.py
import copy
import warnings
import weakref
from itertools import chain
from collections import namedtuple
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Optional, Union, Dict, Any, List, Tuple, Iterable, NamedTuple, Callable

import torch
from torch import Tensor
from torch.cuda import Stream
from torch_sparse import SparseTensor, coalesce
from torch_geometric.utils import contains_isolated_nodes, is_undirected, contains_self_loops

from .typing import (
    NODEKEYS,
    NODEWORLD,
    EDGEKEYS,
    EDGEWORLD,
    CVKEYS,
    CVWORLD,
    DataType,
    NodeType,
    EdgeType,
    CVType,
)
from .data_container import DataContainer
from .mappingview import KeysView, ValuesView, ItemsView
from .torchsparse_patcher import (
    spt_patcher,
    recursive_apply_contiguous,
    recursive_apply_cudastream,
    recursive_apply_get_device
)


NKEYS = {"x", "pos", "node_attr", "batch", "node_feature"}

class BaseStorage(MutableMapping):
    """
    The base is referenced from PyTorch Geometric:
        referenced code: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/storage.py
    This class wraps a Python dictionary and extends it as follows:
    1. It allows attribute assignments, e.g.:
       `storage.x = ...` in addition to `storage['x'] = ...`
    2. It allows private attributes that are not exposed to the user, e.g.:
       `storage._{key} = ...` and accessible via `storage._{key}`
    3. It holds an (optional) weak reference to its parent object, e.g.:
       `storage._parent = weakref.ref(parent)`
    4. It allows iterating over only a subset of keys, e.g.:
       `storage.values('x', 'y')` or `storage.items('x', 'y')
    5. It adds additional PyTorch Tensor functionality, e.g.:
       `storage.cpu()`, `storage.cuda()` or `storage.share_memory_()`
    6. It allows input data is wrapped by DataContainer (DC),
        allowing compatible between graph data and structured data such as image in cv.
        The DC has attributes `is_stacked` and `cpu_only`, allowing img batchsize-wise stacked,
        graph node-wise stacked and upload to gpu specifically
    7. It allows any meta data and unknown data on cpu to be saved in this storage by setattr `metastore`.
    Nota that metastore must be key of input dict, not value dict key such {'x': {'meta'}}. This is not allowed.
        Any cpu data or unknow data must be in metastore, and other key in the input data must be tensor
    """
    def __init__(
            self,
            _mapping: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        super(BaseStorage, self).__init__()
        self._mapping = {}
        # metastore is BaseStorage attribute.
        self.metastore = {}
        for k, v in chain((_mapping or {}).items(), kwargs.items()):
            setattr(self, k, v)

    def _datatype_judge(
            self,
            key:str,
    ) -> DataType:
        assert isinstance(key, str), "input key must be string"
        if key in NODEWORLD:
            return DataType.NODE
        elif key in EDGEWORLD:
            return DataType.EDGE
        elif key in CVWORLD:
            return DataType.CV
        elif any([k in key for k in NODEKEYS]):
            return DataType.NODE
        elif any([k in key for k in EDGEKEYS]):
            return DataType.EDGE
        elif any([k in key for k in CVKEYS]):
            return DataType.CV
        else:
            return DataType.META

    def  _patch_spt(self, data: Union[DataContainer]) -> DataContainer:
        if isinstance(data.data, SparseTensor) and not hasattr(data.data, "spt_apply"):
            data._data = spt_patcher(data.data)
        return data

    def _datacontainer_assignment(
            self,
            key: str,
            value: Union[DataContainer, Tensor, SparseTensor],
            is_graph: bool = True,
    ) -> DataContainer:
        if isinstance(value, DataContainer):
            # in default, the input data is taken care of users
            # there should be no error in the attr.
            if key == 'img' and not value.stack:
                warnings.warn("`img` must be stack data. Backend will automatically set stack = True.")
                value._stack = True
        elif isinstance(value, (Tensor, SparseTensor)):
            # there must be Tensor or SparseTensor
            # in default, the data save in `_mapping` will be standard graph data
            # with stack is False for node-wise stacked, cpu_only is False for uploading gpu.
            value = DataContainer(
                data = value,
                stack = False,
                cpu_only = False,
                is_graph = is_graph,
            )

        if not value.is_tensor_or_sptensor:
            raise ValueError(f"The input key `{key}` must be tensor or SparseTensor, "
                             "except `metastore` key (Optional).")
        return self._patch_spt(value)

    @property
    def _key(self) -> Any:
        return None

    def __len__(self) -> int:
        """The length of the dict contents, metastore is not considered."""
        return len(self._mapping)

    def __setattr__(
            self,
            key: str,
            value: Any,
    ):
        if key == '_parent':
            self.__dict__[key] = weakref.ref(value)
        elif key == 'metastore':
            if not isinstance(value, Mapping):
                raise ValueError("The attr `metastore` key must be dict, "
                                 f"but got {type(value)}")
            self.__dict__[key] = value
        elif key[:1] == '_':
            # "_batch_ptr", "_key"
            self.__dict__[key] = value
        elif isinstance(value, DataContainer):
            value = self._datacontainer_assignment(key, value)
            self[key] = value
        else:
            # we keep setattr alive, compatible between metastore key-value setting and BaseStorage dict content setting
            # Note that unknown key will be thrown into `metastore`, like trash.
            datatype = self._datatype_judge(key)
            if datatype == DataType.META:
                warnings.warn(f"Invalid storage for metadata key `{key}`, "
                              f"instead direct metastore storage is suggested.")
                self.metastore[key] = value
            else:
                is_graph = True if datatype in [DataType.NODE, DataType.EDGE] else False
                value = self._datacontainer_assignment(key, value, is_graph)
                self[key] = value

    def __getattr__(self, key:str) -> Union[Mapping, DataContainer]:
        if key == '_mapping':
            self._mapping = {}
            return self._mapping
        elif key == 'metastore':
            self.metastore = {}
            return self.metastore
        try:
            # we return DataContainer to keep info complete.
            return self[key]
        except KeyError:
            raise AttributeError(
                f":obj:'{self.__class__.__name__}' has no attribute '{key}'"
            )

    def __delattr__(self, key: str):
        if key[:1] == "_":
            del self.__dict__[key]
        else:
            # del attributes from _mapping
            del self[key]
        # how to del the key-value pair in metastore? Just del self.metastore is fine
        # metastore is open, and easy to get.

    def __setitem__(
            self,
            key: str,
            value: Union[DataContainer, Tensor, SparseTensor, None],
    ):
        """This method allows save DataContainer, Tensor, SparseTensor in _mapping"""
        if isinstance(value, DataContainer):
            value = self._datacontainer_assignment(key, value)
            self._mapping[key] = value
        else:
            datatype = self._datatype_judge(key)
            is_graph = True if datatype in [DataType.NODE, DataType.EDGE] else False
            if datatype == DataType.META:
                raise KeyError("Reasonable key must be important for _mapping content setting."
                               f"Any metastore key please move on the BaseStorage.metastore for setting")
            if value is None and key in self._mapping:
                del self._mapping[key]
            elif value is not None:
                # allow _"{key words}" to be saved in _mapping
                value = self._datacontainer_assignment(key, value, is_graph)
                self._mapping[key] = value
            # if value is None and key not in self._mapping, just ignore it.

    def __getitem__(self, key: str) -> DataContainer:
        return self._mapping[key]

    def __delitem__(self, key: str):
        if key in self._mapping:
            del self._mapping[key]

    def __iter__(self) -> Iterable:
        return iter(self._mapping)

    def __copy__(self):
        cp = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            cp.__dict__[k] = v
        cp._mapping = copy.copy(cp._mapping)
        cp.metastore = copy.copy(cp.metastore)
        return cp

    def __deepcopy__(self, memodict):
        cp = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            cp.__dict__[k] = v
        cp._mapping = copy.deepcopy(cp._mapping, memodict)
        cp.metastore = copy.deepcopy(cp.metastore, memodict)
        return cp

    def __getstate__(self) -> Dict[str, Any]:
        cp = self.__dict__.copy()
        _parent = cp.get('_parent', None)

        if _parent is not None:
            cp['_parent'] = _parent()
        return cp

    def __setstate__(self, mapping: Dict[str, Any]):
        for k, v in mapping.items():
            self.__dict__[k] = v

        _parent = self.__dict__.get('_parent', None)
        if _parent is not None:
            self.__dict__['_parent'] = weakref.ref(_parent)

    def __repr__(self):
        """Return _mapping rather than metastore"""
        return repr(self._mapping)

    # Allow iterating over subsets in mapping keys, values, items
    # In contrast to PyTorch Geometric, we implement _mapping and metastore
    # view, and :cls:keys, values, items mainly focus on _mapping
    def mapping_keys(self, *args: List[str]) -> KeysView:
        return KeysView(self._mapping, *args)

    def mapping_values(self, *args: List[str]) -> ValuesView:
        return ValuesView(self._mapping, *args)

    def mapping_items(self, *args: List[str]) -> ItemsView:
        return ItemsView(self._mapping, *args)

    def metastore_keys(self, *args: List[str]) -> KeysView:
        return KeysView(self.metastore, *args)

    def metastore_values(self, *args: List[str]) -> ValuesView:
        return ValuesView(self.metastore, *args)

    def metastore_items(self, *args: List[str]) -> ItemsView:
        return ItemsView(self.metastore, *args)

    def keys(self, *args: List[str]) -> KeysView:
        return self.mapping_keys(*args)

    def values(self, *args: List[str]) -> ValuesView:
        return self.mapping_values(*args)

    def items(self, *args: List[str]) -> ItemsView:
        return self.mapping_items(*args)

    # data transform
    def to_dict(
            self,
            decode: bool = False,
            drop_meta: bool = False,
    ) -> Dict[str, Any]:
        """
        Return the dictionary of stored key-value pair,
        metastore data will be ignored.
        """
        mapping = copy.copy(self._mapping)
        if decode:
            for k, v in mapping.items():
                mapping[k] = v.data
        if len(self.metastore) > 0 and not drop_meta:
            mapping['metastore'] = copy.copy(self.metastore)
        return mapping

    def to_namedtuple(self, decode: bool = False) -> NamedTuple:
        """
        Returns the namedtuple to store key-value pair,
        metastore data will be ignored.
        """
        field = list(self.keys())
        tuplename = f"{self.__class__.__name__}Tuple"
        StorageTuple = namedtuple(tuplename, field)
        return StorageTuple(*[self[k].data if decode else self[k] for k in field])

    def apply_(self, func: Callable, *args: List[str]):
        """
        Apply the in-place function :Callable: `func` to either all attributes
            or the ones of `*args`
        """
        for v in self.values(*args):
            assert isinstance(v, DataContainer)
            recursive_apply_(v.data, func)

    def apply(
            self,
            func: Callable,
            *args: List[str],
    ):
        """
        Apply function :Callable: `func` to either all attributes
            or the ones of `*args`
        """
        for k, v in self.items(*args):
            assert isinstance(v, DataContainer)
            data = recursive_apply(v.data, func)
            self[k].update_(data)
        return self

    # pytorch tensor functionality
    def clone(self, *args: List[str]):
        """Apply deep-copy of the object"""
        return copy.deepcopy(self)

    def contiguous(self, *args: List[str]):
        """
        Ensure a contiguous memory layout, either all attributes
            or the ones of `*args`
        """
        for k, v in self.items(*args):
            assert isinstance(v, DataContainer)
            if not v.is_sptensor:
                data = recursive_apply(v.data, lambda x: x.contiguous())
            else:
                data = recursive_apply_contiguous(v.data)
            self[k].update_(data)

    def pin_memory(self, *args: List[str]):
        """
        Copy attibutes to pinned memory, either all attributes
            or the ones of `*args`
        """
        return self.apply(lambda x: x.pin_memory(), *args)

    def share_memory(self, *args: List[str]):
        """
        Moves attributes to shared memory, either for all attributes
            or the ones of `*args`
        """
        return self.apply(lambda x: x.share_memory_(), *args)

    def detach(self, *args: List[str]):
        """
        Detaches attributes from the computation graph by creating a new
            tensor, either for all attributes or the ones of `*args`
        """
        return self.apply(lambda x: x.detach(), *args)

    def detach_(self, *args: List[str]):
        """
        Detaches attributes from the computation graph by the manner of in-plane,
            either for all attributes or the ones of `*args`
        """
        return self.apply(lambda x: x.detach_(), *args)

    def requires_grad_(self, *args: List[str], requires_grad: bool = True):
        """Tracks gradient computation, either for all attributes or the ones of `*args`"""
        return self.apply(
            lambda x: (x.requires_grad_(requires_grad = requires_grad) if x.is_floating_point() else x),
            *args)

    def cpu(self, *args: List[str]):
        """Copy attributes to CPU memory, either for all attributes or the ones of `*args`"""
        return self.apply(lambda x: x.cpu(), *args)

    def _cuda_filter_apply(self, func: Callable,*args: List[str]):
        """
        Helper function: filte the :obj:DataContainer by the `cpu_only` attribute,
        blocking the cpu data in dictionary.
        """
        for k, v in self.items(*args):
            assert isinstance(v, DataContainer)
            if not v.cpu_only:
                data = recursive_apply(v.data, func)
                self[k].update_(data)
        return self

    def to(self, device: Union[str, int], *args: List[str],
           non_blocking: bool = True):
        """Apply device conversion, either for all attributes or the ones of `*args`"""
        if device == 'cpu' or device == -1:
            self.apply(lambda x: x.to(device = device, non_blocking = non_blocking), *args)
        elif 'cuda' in device or device > -1:
            self._cuda_filter_apply(lambda x: x.to(device = device, non_blocking = non_blocking), *args)

    def cuda(self, device: Union[str, int], *args: List[str],
           non_blocking: bool = True):
        """Copy attributes to CUDA memory, either for all attributes or the ones of `*args` with cpu_only = False"""
        return self._cuda_filter_apply(lambda x: x.cuda(device = device, non_blocking = non_blocking), *args)

    def record_stream(self, stream: Stream, *args: List[str]):
        """Ensure tensor memory is not reused until work on main stream is complete"""
        for k, v in self.items(*args):
            assert isinstance(v, DataContainer)
            if not v.cpu_only:
                if not v.is_sptensor:
                    recursive_apply_(v.data, lambda x: x.record_stream(stream))
                else:
                    recursive_apply_cudastream(v.data, stream)

    def is_cuda(self, *args: List[str]):
        """
        Judege attributes on CUDA, either for all attributes or the ones of `*args` with cpu_only = False.
        Once there are specified attributes in CUDA, it means all `cpu_only = False` attributes will be in CUDA.
        """
        for v in self.values(*args):
            flag = v.data.is_cuda() if v.is_sptensor else v.data.is_cuda
            if flag:
                return True
        return False

    def get_device(self):
        """Get tensor or sparsetensor device. either for all attributes or the ones of `*args` with cpu_only = False"""
        for v in self.values():
            if v.is_tensor_or_sptensor:
                # deal with some data with cpu_only = True
                device = recursive_apply_get_device(v.data)
                if device != -1:
                    return device
        return -1


class NodeStorage(BaseStorage):
    """
    :obj:NodeStorage supports graph node data, and has been perfected by PyTorch Geometric team.
    Referenced code: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/storage.py
    """
    @property
    def _key(self) -> NodeType:
        """Hetero data required node attributes"""
        key = self.__dict__.get('_key', None)
        if key is None or not isinstance(key, str):
            raise ValueError('Attribute `_key` is invalid, it must be `NodeType` (string)')
        return key

    @property
    def can_infer_num_nodes(self) -> bool:
        keys = set(self.keys())
        num_node_keys = {
            'num_nodes', 'x', 'pos', 'batch', 'adj', 'adj_t', 'edge_index', 'face'
        }
        if len(keys & num_node_keys) > 0:
            return True
        elif len([key for key in keys if 'node' in key]) > 0:
            return True
        else:
            return False

    @property
    def num_nodes(self) -> Optional[int]:
        # Sequential access attributes that reveal the number of nodes
        # `num_nodes` is not allowed to be saved in BaseStorage dict.
        if 'num_nodes' in self.metastore:
            return self.metastore['num_nodes']
        for key, value in self.items():
            if value.is_tensor and (key in NKEYS or any(k in key for k in NODEKEYS)):
                cat_dim = self._parent().__cat_dim__(key, value, self)
                if cat_dim is None:
                    continue
                return value.size(cat_dim)
        if 'adj' in self and self['adj'].is_sptensor:
            return self['adj'].size(0)
        if 'adj_t' in self and self['adj_t'].is_sptensor:
            return self['adj_t'].size(1)
        warnings.warn(
            f"Unable to accurately infer 'num_nodes' from the attribute set "
            f"'{set(self.keys())}'. Please explicitly set 'num_nodes' as an "
            f"attribute of " +
            ("'data'" if self._key is None else f"'data[{self._key}]'") +
            " in `metastore` to suppress this warning")
        # the below simple infer mechanism is sometimes unreasonable and dangerous
        # especially when graph has isolated nodes.
        if 'edge_index' in self and self['edge_index'].is_tensor:
            if self['edge_index'].numel() > 0:
                return int(self['edge_index'].max) + 1
            else:
                return 0
        if 'face' in self and self['face'].is_tensor:
            if self['face'].numel() > 0:
                return int(self['face'].max) + 1
            else:
                return 0
        return None

    @property
    def num_node_features(self) -> int:
        if 'x' in self and self['x'].is_tensor_or_sptensor:
            return 1 if self.x.dim() == 1 else self.x.size(-1)
        if 'node_feature' in self and self['node_feature'].is_tensor_or_sptensor:
            return 1 if self.node_feature.dim() == 1 else self.node_feature.size(-1)
        if 'node_attr' in self and self['node_attr.data'].is_tensor_or_sptensor:
            return 1 if self.node_attr.dim() == 1 else self.node_attr.size(-1)
        return 0

    @property
    def num_features(self) -> int:
        return self.num_node_features

    def is_node_attr(self, key: str) -> bool:
        value = self[key]
        cat_dim = self._parent().__cat_dim__(key, value, self)
        if cat_dim is None:
            return False
        if not value.is_tensor:
            return False
        if value.dim() == 0 or value.size(cat_dim) != self.num_nodes:
            return False
        return True

    def is_edge_attr(self, key: str) -> bool:
        return False

    def is_cv_attr(self, key: str) -> bool:
        return False

class EdgeStorage(BaseStorage):
    """
    :obj:EdgeStorage supports graph edge data, and has been perfected by PyTorch Geometric team.
    Referenced code: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/storage.py
    PyG team support multiple ways to store edge connectivity in a
    :class:`EdgeStorage` object:
    * :obj:`edge_index`: A :class:`torch.LongTensor` holding edge indices in
      COO format with shape :obj:`[2, num_edges]` (the default format)
    * :obj:`adj`: A :class:`torch_sparse.SparseTensor` holding edge indices in
      a sparse format, supporting both COO and CSR format.
    * :obj:`adj_t`: A **transposed** :class:`torch_sparse.SparseTensor` holding
      edge indices in a sparse format, supporting both COO and CSR format.
      This is the most efficient one for graph-based deep learning models as
      indices are sorted based on target nodes.
    we follow pyg's implementation and use pyg's utils library to support the data transform operation
        in our modified EdgeStorage.
    """
    @property
    def _key(self) -> EdgeType:
        """Hetero data required edge attributes"""
        key = self.__dict__.get('_key', None)
        if key is None or not isinstance(key, tuple) or not len(key):
            raise ValueError('Attribute `_key` is invalid, it must be `EdgeType` (Tuple(str, str, str))')
        return key

    @property
    def edge_index(self) -> DataContainer:
        if 'edge_index' in self:
            # ip identity
            return self['edge_index']
        if 'adj' in self:
            # adj: row, col
            adj = self['adj']
            data = torch.stack(adj.data.coo[:2], dim = 0)
            # hetero ip
            return adj.update(data)
        if 'adj_t' in self:
            # adj: col, row
            adj_t = self['adj_t']
            data = torch.stack(adj_t.data.coo[-2::-1], dim = 0)
            # hetero ip
            return adj_t.update(data)
        raise AttributeError(f":cls: {self.__class__.__name__} can not get attribute `edge_index`, "
                             f"as there are no `edge_index`, `adj`, `adj_t`.")

    @property
    def num_edges(self) -> int:
        for k, v in self.items():
            # this will filter adj and adj_t
            if v.is_tensor and "edge" in k:
                cat_dim = self._parent().__cat_dim__(k, v, self)
                if cat_dim is None:
                    continue
                return v.size(cat_dim)
        # we implement the other iteration for adj and adj_t, rather than the above
        # because other data maybe SparseTensor
        for v in self.values('adj', 'adj_t'):
            v = v.data
            if isinstance(v, SparseTensor):
                return v.nnz()
        # this means there is no edge
        return 0

    @property
    def num_edge_features(self) -> int:
        if 'edge_attr' in self and self['edge_attr'].is_tensor:
            return 1 if self['edge_attr'].dim() == 1 else self['edge_attr'].size(-1)
        if 'edge_feature' in self and self['edge_feature'].is_tensor:
            return 1 if self['edge_feature'].dim() == 1 else self['edge_feature'].size(-1)
        return 0

    @property
    def num_features(self) -> int:
        return self.num_edge_features

    def is_node_attr(self, key: str) -> bool:
        return False

    def is_edge_attr(self, key: str) -> bool:
        value = self[key]
        if not value.is_tensor:
            return False
        cat_dim = self._parent().__cat_dim__(key, value, self)
        if value.dim() == 0 or value.size(cat_dim) !=  self.num_edges:
            return False
        return True

    def is_cv_attr(self, key: str) -> bool:
        return False

    def size(
            self,
            dim: Optional[int] = None,
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        key = self._key
        if key is None:
            # when Data call, it always turn out to raise error
            raise NameError("Unable to infer size without explict `_key` assignment.")

        size = (
            self._parent()[key[0]].num_nodes,
            self._parent()[key[-1]].num_nodes
        )
        return size if dim is None else size[dim]

    def is_coalesced(self) -> bool:
        for v in self.values('adj', 'adj_t'):
            return v.data.is_coalesced()
        edge_index = self.edge_index.data
        coalesce_edge_index, _ = coalesce(edge_index, None, self.size(0), self.size(1))
        return (edge_index.numel() == coalesce_edge_index.numel() and
                bool((edge_index == coalesce_edge_index).all()))

    def coalesced(
            self,
            reduce: str = 'sum',
    ):
        for k, v in self.items('adj', 'adj_t'):
            self[v].update_(v.data.coalesce(reduce))

        if 'edge_index' in self:
            edge_index = self.edge_index.data
            if 'edge_attr' in self:
                key = 'edge_attr'
                edge_attr = self.edge_attr.data
            elif 'edge_feature' in self:
                key = 'edge_feature'
                edge_attr = self.edge_feature.data
            else:
                key = None
                edge_attr = None
            edge_index, edge_attr = coalesce(
                edge_index, edge_attr,
                *self.size(), op = reduce
            )
            self['edge_index'].update_(edge_index)
            if edge_attr is not None:
                self[key].update_(edge_attr)
        return self

    def has_isolated_nodes(self) -> bool:
        edge_index, num_nodes = self.edge_index.data, self.size(1)
        if num_nodes is None:
            raise NameError("Unable to infer 'num_nodes'")
        if self.is_bipartite():
            return torch.unique(edge_index[1]).numel() < num_nodes
        else:
            return contains_isolated_nodes(edge_index, num_nodes)

    def has_self_loops(self) -> bool:
        if self.is_bipartite():
            return False
        edge_index = self.edge_index.data
        return contains_self_loops(edge_index)

    def is_undirected(self) -> bool:
        # hetero part
        if self.is_bipartite():
            return False
        # homogenous part
        for v in self.values('adj', 'adj_t'):
            return v.data.is_symmetric()
        edge_index = self.edge_index.data
        if 'edge_attr' in self:
            edge_attr = self.edge_attr.data
        elif 'edge_feature' in self:
            edge_attr = self.edge_feature.data
        else:
            edge_attr = None
        return is_undirected(edge_index, edge_attr, num_nodes = self.size(0))

    def is_directed(self) -> bool:
        return not self.is_undirected()

    def is_bipartite(self) -> bool:
        """
        Base Data (homogenous data) is not bipartite,
            it must be hetero data and the source node and target node is different.
        """
        return self._key is not None and self._key[0] != self._key[-1]



class CVStorage(BaseStorage):

    @property
    def _key(self) -> CVType:
        key = self.__dict__.get('_key', None)
        if key is None or not isinstance(key, str):
            raise ValueError('Attribute `_key` is invalid, it must be CVType (string)')
        return key

    @property
    def channels(self) -> Optional[int]:
        """Default img shape [C, H, W]"""
        assert 'img' in self, "attr `img` must assign."
        size = self['img'].size()
        dim = self['img'].dim()
        # single rgb img (3, H, W), gray-scale img (1, H, W), img feature (C, H, W)
        # or batched img (B, C, H, W)
        if dim == 3 or dim == 4:
            return size[-3]
        # squeezed gray-scale img (H, W)
        if dim == 2:
            return 1
        # other dim will be ignored
        return None

    @property
    def num_features(self) -> int:
        """Detail see :method:`channels`"""
        return self.channels

    def cvsize(
            self,
            dim: Optional[int] = None,
    ) -> Union[torch.Size, int, None]:
        """If there is cv task, there must have attr `img`, otherwise return None"""
        if 'img' in self:
            size = self['img'].size()
            return size if dim is None else size[dim]
        else:
            return None

    def size(
            self,
            dim: Optional[int] = None,
    ) -> Union[Tuple[int], int, None]:
        assert 'img' in self, "attr `img` must assign."
        size = self['img'].size()
        dim = self['img'].dim()
        # single rgb img (3, H, W), gray-scale img (1, H, W), img feature (C, H, W)
        # or batched img (B, C, H, W)
        if dim == 3 or dim == 4:
            size = tuple(size[-2:])
            return size if dim is None else size[dim]
        # squeezed gray-scale img (H, W)
        if dim == 2:
            return tuple(size) if dim is None else size[dim]
        # other dim will be ignored
        return None

    def is_node_attr(self, key: str) -> bool:
        return False

    def is_edge_attr(self, key: str) -> bool:
        return False

    def is_cv_attr(self, key: str) -> bool:
        value = self[key]
        if not value.is_tensor:
            return False
        cat_dim = self._parent().__cat_dim__(key, value, self)
        # img data needs stack
        if cat_dim is None:
            return True
        if key in CVWORLD or any(k in key for k in CVKEYS):
            return True
        # batched img or (C, H, W)-shape img
        if value.dim() in [3, 4]:
            return True
        return False


class GlobalStorage(NodeStorage, EdgeStorage, CVStorage):
    """
    The GlobalStorage is used to base :obj: Data, saving graph data
        and cv img data, etc
    """
    @property
    def _key(self):
        """Base :obj: Data (homogenous data) use this override method"""
        return None

    def size(
            self,
            dim: Optional[int] = None,
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        """source node and target node in Data come from the sam node matrix"""
        size = (self.num_nodes, self.num_nodes)
        return size if dim is None else size[dim]

    @property
    def num_features(self) -> int:
        return self.num_node_features

    def is_node_attr(self, key: str) -> bool:
        value = self[key]
        if not value.is_tensor:
            return False
        cat_dim = self._parent().__cat_dim__(key, value, self)
        if cat_dim is None:
            return False
        # in default, we think of cat_dim = None is img data and
        # graph node data has not this behavior
        num_nodes, num_edges = self.num_nodes, self.num_edges
        if value.dim() == 0 or value.size(cat_dim) != num_nodes:
            return False
        if key in EDGEWORLD or any(k in key for k in EDGEKEYS):
            return False
        return num_nodes == num_edges

    def is_edge_attr(self, key: str) -> bool:
        value = self[key]
        if not value.is_tensor:
            return False
        cat_dim = self._parent().__cat_dim__(key, value, self)
        # in default, we think of cat_dim = None is img data and
        # graph edge data has not this behavior
        if cat_dim is None:
            return False
        num_nodes, num_edges = self.num_nodes, self.num_edges
        if value.dim() == 0 or value.size(cat_dim) != num_edges:
            return False
        if key in NODEWORLD or any(k in key for k in NODEKEYS):
            return False
        return num_nodes == num_edges

    def is_cv_attr(self, key: str) -> bool:
        return super(EdgeStorage, self).is_cv_attr(key)



def recursive_apply_(
        data: Any,
        func: Callable,
):
    """
    Recursive apply function to data, including Tensor, namedtyple, Sequence but not string, Mapping.
        Note that SparseTensor does not supports record_stream, contiguous function.
    Referenced code: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/storage.py
    """
    if isinstance(data, Tensor):
        func(data)
    elif isinstance(data, tuple) and hasattr(data, '_field'):  # namedtuple
        for d in data:
            recursive_apply_(d, func)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        for d in data:
            recursive_apply_(d, func)
    elif isinstance(data, Mapping):
        for v in data.values():
            recursive_apply_(v, func)
    else:
        # usually SparseTensor has problems, no record_stream and contiguous function
        try:
            func(data)
        except:
            pass

def recursive_apply(
        data: Any,
        func: Callable,
) -> Any:
    """
    Recursive apply function to data, including Tensor, namedtyple, Sequence but not string, Mapping, rnn.PackedSequence
        Note that SparseTensor does not supports record_stream, contiguous function.
    Referenced code: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/storage.py
    """
    if isinstance(data, Tensor):
        return func(data)
    # experimental code
    # try to apply tensor func to SparTensor.
    elif isinstance(data, SparseTensor):
        try:
            return func(data)
        except Exception as e:
            print(f"Error: {e}")
            return data
    elif isinstance(data, torch.nn.utils.rnn.PackedSequence):
        return func(data)
    elif isinstance(data, tuple) and hasattr(data, '_field'):  # namedtuple
        return type(data)(*(recursive_apply(d, func) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [recursive_apply(d, func) for d in data]
    elif isinstance(data, Mapping):
        return {k: recursive_apply(v, func) for k, v in data.items()}
    else:
        # usually SparseTensor has problems, no record_stream and contiguous function
        try:
            return func(data)
        except:
            return data