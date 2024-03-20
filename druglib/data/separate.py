# Copyright (c) MDLDrugLib. All rights reserved.
from collections.abc import Mapping, Sequence
from typing import Any

from .data import BaseData
from .storage import BaseStorage
from .data_container import DataContainer


def separate(
        cls,
        batch: BaseData,
        idx: int,
        slice_dict: Any,
        inc_dict: Any = None,
        decrement: bool = True,
) -> BaseData:
    # Separates the individual element from a `batch` at index `idx`.
    # `separate` can handle both homogeneous and heterogeneous data objects by
    # individually separating all their stores.
    # In addition, `separate` can handle nested data structures such as
    # dictionaries and lists.

    data = cls().stores_as(batch)

    # We iterate over each storage object and recursively separate all its
    # attributes:
    for batch_store, data_store in zip(batch.stores, data.stores):
        # The `num_nodes` attribute needs special treatment, as we cannot infer
        # the real number of nodes from the total number of nodes alone:
        num_nodes_list = batch_store.metastore.pop('_num_nodes', None)
        batch_store.metastore.pop('num_nodes', None)
        data_store.metastore = _metastore_separate(batch_store.metastore, idx)
        if num_nodes_list is not None:
            data_store.metastore['num_nodes'] = num_nodes_list[idx]
        find_keys = None
        for k in data_store.metastore.keys():
            if '_SPT_' in k:
                find_keys = k
                break
        if find_keys:
            # find SparseTensor formatting of edge_index saved in metastore
            data_store[find_keys[5:]] = DataContainer(
                data = data_store.metastore.pop(find_keys), is_graph = True
            )

        key = batch_store._key
        if key is not None:
            slice_dict[key].pop(find_keys, None)
            attrs = slice_dict[key].keys()
        else:
            slice_dict.pop(find_keys, None)
            attrs = set(batch_store.keys())
            attrs = [attr for attr in slice_dict.keys() if attr in attrs]
        for attr in attrs:
            if key is not None:
                slices = slice_dict[key][attr]
                incs = inc_dict[key][attr] if decrement else None
            else:
                slices = slice_dict[attr]
                incs = inc_dict[attr] if decrement else None
            data_store[attr] = _data_separate(attr, batch_store[attr], idx, slices,
                                         incs, batch, batch_store, decrement)

    return data


def _data_separate(
    key: str,
    template: DataContainer,
    idx: int,
    slices: Any,
    incs: Any,
    batch: BaseData,
    store: BaseStorage,
    decrement: bool,
) -> Any:

    if template.is_tensor:
        # Narrow a `torch.Tensor` based on `slices`.
        # NOTE: We need to take care of decrementing elements appropriately.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, template, store)
        start, end = int(slices[idx]), int(slices[idx + 1])
        value = template.data
        value = value.narrow(cat_dim or 0, start, end - start)
        value = value.squeeze(0) if cat_dim is None else value
        if decrement and (incs.dim() > 1 or int(incs[idx]) != 0):
            value = value - incs[idx].to(value.device)
        return value

    elif template.is_sptensor and decrement:
        # Narrow a `SparseTensor` based on `slices`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, template, store)
        cat_dims = (cat_dim, ) if isinstance(cat_dim, int) else cat_dim
        value = template.data
        for i, dim in enumerate(cat_dims):
            start, end = int(slices[idx][i]), int(slices[idx + 1][i])
            value = value.narrow(dim, start, end - start)
        return value

    else:
        # In theory, this condition will not be used
        return template[idx]

def _metastore_separate(
    value: Any,
    idx: int,
) -> Any:
    if isinstance(value, Mapping):
        # Recursively separate elements of dictionaries.
        return {
            key: _metastore_separate(elem, idx)
            for key, elem in value.items()
        }
    else:
        return value[idx]