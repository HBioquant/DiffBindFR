# Copyright (c) MDLDrugLib. All rights reserved.
# Modified from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/collate.py
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch_sparse import SparseTensor, cat

from .data_container import DataContainer
from .data import BaseData
from .storage import BaseStorage, NodeStorage


def data_collate(
    cls,
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Optional[Union[List[str]]] = None,
    exclude_keys: Optional[Union[List[str]]] = None,
) -> Tuple[BaseData, Mapping, Mapping]:
    """
    Collates a list of `data` objects into a single object of type `cls`.
    `collate` can handle both homogeneous and heterogeneous data objects by
        individually collating all their stores.
    In addition, `collate` can handle nested data structures such as dictionaries and lists.
    Referenced Code: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/collate.py
    We modified the code so it will be available to DataContainer version :cls:`Data`, and collates image data
        with batchsize-wise batching.
    """
    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:
        out = cls(_base_cls=data_list[0].__class__)  # Dynamic inheritance.
    else:
        out = cls()

    # Create empty stores:
    out.stores_as(data_list[0])

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_store_list = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    device = None
    slice_dict, inc_dict = defaultdict(dict), defaultdict(dict)
    for out_store in out.stores:
        # 1. get the batch size of storages by key.
        key = out_store._key
        stores = key_to_stores[key]

        # 2. `metastore` needs individual treatment
        meta = [store.metastore for store in stores]
        meta = _meta_collate(meta)
        # The `num_nodes` attribute needs special treatment, as we need to
        # sum their values up instead of merging them to a list:
        num_nodes = meta.get("num_nodes", None)
        if num_nodes is not None:
            meta["num_nodes"] = sum(num_nodes)
            meta["_num_nodes"] = num_nodes
        out_store.metastore = meta

        # 3. various attributes treatment
        for attr in stores[0].keys():
            # Do not include top-level attribute.
            if attr in exclude_keys:
                continue
            # Skip batching of `ptr` vectors for now:
            if attr == 'ptr':
                continue

            # DataContainer template
            template: DataContainer = stores[0][attr]

            # get the batch size value of _key.attr
            values = [store[attr].data for store in stores]

            # Collate attributes (Tensor, SparseTensor) into a unified representation:
            value, slices, incs = _data_collate(
                attr, values, data_list, stores, template, increment
            )

            if isinstance(value, Tensor) and value.is_cuda:
                device = value.device
            if isinstance(value, list):
                # Usually, when increment set to False,
                # the SparseTensor will be saved in list
                # we instead save it in metastore
                out_store.metastore["_SPT_" + attr] = value
            else:
                out_store[attr] = template.update(value)
            if key is not None:
                slice_dict[key][attr] = slices
                inc_dict[key][attr] = incs
            else:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute:
            if attr in follow_batch:
                batch, ptr = _batch_and_ptr(slices, device)
                out_store[f'{attr}_batch'] = DataContainer(
                    batch, stack = False, cpu_only = template.cpu_only)
                out_store[f'{attr}_ptr'] = DataContainer(
                    ptr, stack = False, cpu_only = template.cpu_only)

        # In case the storage holds node, we add a top-level batch vector it:
        if (add_batch and isinstance(stores[0], NodeStorage)
                and stores[0].can_infer_num_nodes):
            repeats = [store.num_nodes for store in stores]
            out_store.batch = repeat_interleave(repeats, device = device)
            out_store.ptr = cumsum(torch.tensor(repeats, device = device))

    return out, slice_dict, inc_dict

def _data_collate(
    key: str,
    values: List[Any],
    data_list: List[BaseData],
    stores: List[BaseStorage],
    template: DataContainer,
    increment: bool,
) -> Tuple[Any, Any, Any]:

    elem = values[0]

    if isinstance(elem, Tensor):
        # Concatenate a list of `torch.Tensor` along the `cat_dim`.
        # NOTE: We need to take care of incrementing elements appropriately.
        key = str(key)
        cat_dim = data_list[0].__cat_dim__(key, template, stores[0])
        if (cat_dim is None) or elem.dim() == 0:
            # this part can handle img data by padding
            values = _data_padding(values, template)

        slices = cumsum([value.size(cat_dim or 0) for value in values])
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if incs.dim() > 1 or int(incs[-1]) != 0:
                values = [
                    value + inc.to(value.device)
                    for value, inc in zip(values, incs)
                ]
        else:
            incs = None

        if torch.utils.data.get_worker_info() is not None:
            # Write directly into shared memory to avoid an extra copy:
            numel = sum(value.numel() for value in values)
            storage = elem.storage()._new_shared(numel)
            shape = list(elem.size())
            if cat_dim is None or elem.dim() == 0:
                shape = [len(values)] + shape
            else:
                shape[cat_dim] = int(slices[-1])
            out = elem.new(storage).resize_(*shape)
        else:
            out = None

        value = torch.cat(values, dim=cat_dim or 0, out=out)
        return value, slices, incs

    elif isinstance(elem, SparseTensor) and increment:
        # Concatenate a list of `SparseTensor` along the `cat_dim`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        key = str(key)
        cat_dim = data_list[0].__cat_dim__(key, template, stores[0])
        cat_dims = (cat_dim, ) if isinstance(cat_dim, int) else cat_dim
        repeats = [[value.size(dim) for dim in cat_dims] for value in values]
        slices = cumsum(repeats)
        value = cat(values, dim=cat_dim)
        return value, slices, None
    else:
        assert isinstance(elem, SparseTensor), "All data must be Tensor or SparseTensor in `Storage` dict"
        # Other-wise, just return the list of values as it is.
        # Normally, element must be tensor.
        slices = torch.arange(len(values) + 1)
        return values, slices, None

def _meta_collate(
    values: List[Any],
) -> Any:
    elem = values[0]
    if isinstance(elem, Mapping):
        # Recursively collate elements of dictionaries.
        value_dict = {}
        for key in elem.keys():
            value_dict[key] = _meta_collate([v[key] for v in values])
        return value_dict
    else:
        return values

def _data_padding(
        batch: List[Tensor],
        template: DataContainer,
) -> List[Tensor]:
    """
    Template on the first element of DataContainer, padding all values for the same shape
        referenced by max side, thus allowing batchsize-wise batching.
    Typically, input img (shape [C, H, W]) return tensor (shape [1, C', H', W'])
    """
    assert template.is_tensor, "stacked data must be Tensor"
    if template.pad_dims is not None:
        ndim = template.dim()
        pad_dims = template.pad_dims
        assert pad_dims in [None, 1, 2, 3], f'pad_dim is limited to None or 1~3, but got {pad_dims}'
        assert ndim > pad_dims
        # prepare the max shape collection
        max_shape = [0 for _ in range(pad_dims)]
        for dim in range(1, pad_dims + 1):
            max_shape[dim - 1] = template.size(-dim)# firstly set a reference
        # find max shape per axis
        for value in batch:
            # forwardly check unconsidered axis at the same size
            for dim in range(0, ndim - pad_dims):
                assert value.size(dim) == template.size(dim), "Size of unconsidered aixs is inconsistent, " \
                f"get sample size {value.size()} while template size {template.sizes} with attr `pad_dims` = {pad_dims}"
            # reversely get max shape
            for dim in range(1, pad_dims + 1):
                max_shape[dim - 1] = max(max_shape[dim - 1], value.size(-dim))
        padded_batch = []
        for value in batch:
            pad = [0 for _ in range(pad_dims * 2)]
            for dim in range(1, pad_dims + 1):
                pad[2 * dim - 1] = max_shape[dim - 1] - value.size(-dim)# right-pad mode
            padded_batch.append(
                F.pad(
                    value, pad, value = template.padding_value,
                ).unsqueeze(0)
            )
        return padded_batch
    elif template.pad_dims is None:
        return [value.unsqueeze(0) for value in batch]
    else:
        raise ValueError('pad_dims should be either None or Integers (1-3).')


def _batch_and_ptr(
    slices: Any,
    device: Optional[torch.device] = None,
) -> Tuple[Any, Any]:
    if (isinstance(slices, Tensor) and slices.dim() == 1):
        # Default case, turn slices tensor into batch.
        repeats = slices[1:] - slices[:-1]
        batch = repeat_interleave(repeats.tolist(), device=device)
        ptr = cumsum(repeats.to(device))
        return batch, ptr

    elif isinstance(slices, Mapping):
        # Recursively batch elements of dictionaries.
        batch, ptr = {}, {}
        for k, v in slices.items():
            batch[k], ptr[k] = _batch_and_ptr(v, device)
        return batch, ptr

    elif (isinstance(slices, Sequence) and not isinstance(slices, str)
          and isinstance(slices[0], Tensor)):
        # Recursively batch elements of lists.
        batch, ptr = [], []
        for s in slices:
            sub_batch, sub_ptr = _batch_and_ptr(s, device)
            batch.append(sub_batch)
            ptr.append(sub_ptr)
        return batch, ptr

    else:
        # Failure of batching, usually due to slices.dim() != 1
        return None, None


###############################################################################


def repeat_interleave(
    repeats: List[int],
    device: Optional[torch.device] = None,
) -> Tensor:
    outs = [torch.full((n, ), i, device=device) for i, n in enumerate(repeats)]
    return torch.cat(outs, dim=0)


def cumsum(value: Union[Tensor, List[int]]) -> Tensor:
    if not isinstance(value, Tensor):
        value = torch.tensor(value)
    out = value.new_empty((value.size(0) + 1, ) + value.size()[1:])
    out[0] = 0
    torch.cumsum(value, 0, out=out[1:])
    return out


def get_incs(key, values: List[Any], data_list: List[BaseData],
             stores: List[BaseStorage]) -> Tensor:
    repeats = [
        data.__inc__(key, value, store)
        for value, data, store in zip(values, data_list, stores)
    ]
    if isinstance(repeats[0], Tensor):
        repeats = torch.stack(repeats, dim=0)
    else:
        repeats = torch.tensor(repeats)
    return cumsum(repeats[:-1])

