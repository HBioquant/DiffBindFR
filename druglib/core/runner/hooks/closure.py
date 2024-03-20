# Copyright (c) MDLDrugLib. All rights reserved.
from .hook import HOOKS, Hook

# add some method for Hook
@HOOKS.register_module()
class ClosureHook(Hook):

    def __init__(self, fn_name, fn):
        assert hasattr(self, fn_name)
        assert callable(fn)
        setattr(self, fn_name, fn)

@HOOKS.register_module()
class MultiClosureHook(Hook):

    def __init__(self, fnmapping:dict):
        for fn_name, fn in fnmapping.items():
            assert hasattr(self, fn_name)
            assert callable(fn)
            setattr(self, fn_name, fn)