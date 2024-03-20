# Copyright (c) MDLDrugLib. All rights reserved.
# Copied from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/view.py
from typing import (
    Iterable,
    List
)
from collections.abc import Mapping


class MappingView(object):
    __class__getitem__ = classmethod(type([]))

    def __init__(
            self,
            _mapping: Mapping,
            *args: List[str],
    ):
        self._mapping = _mapping
        self._args = args

    def _keys(self) -> Iterable:
        if len(self._args) == 0:
            return self._mapping.keys()
        else:
            return [k for k in self._args if k in self._mapping]

    def __len__(self) -> int:
        return len(self._keys())

    def __repr__(self) -> str:
        mapping = {k: self._mapping[k] for k in self._keys()}
        return f"{self.__class__.__name__}({mapping})"

class KeysView(MappingView):
    def __iter__(self) -> Iterable:
        yield from self._keys()

class ValuesView(MappingView):
    def __iter__(self) -> Iterable:
        for k in self._keys():
            yield self._mapping[k]

class ItemsView(MappingView):
    def __iter__(self) -> Iterable:
        for k in self._keys():
            yield (k, self._mapping[k])