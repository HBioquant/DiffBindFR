# Copyright (c) MDLDrugLib. All rights reserved.

class CastMixin:
    """Copied from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/mixin.py"""
    @classmethod
    def cast(cls, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            elem = args[0]
            if elem is None:
                return None
            if isinstance(elem, CastMixin):
                return elem
            if isinstance(elem, (tuple, list)):
                return cls(*elem)
            if isinstance(elem, dict):
                return cls(**elem)
        return cls(*args, **kwargs)