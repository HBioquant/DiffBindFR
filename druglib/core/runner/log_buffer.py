# Copyright (c) MDLDrugLib. All rights reserved.
from collections import OrderedDict

import numpy as np

class LogBuffer:

    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(
            self,
            vars:dict,
            count:int = 1,
    ):
        assert isinstance(vars, dict)
        for k, v in vars.items():
            if k not in self.val_history:
                self.val_history[k] = []
                self.n_history[k] = []
            self.val_history[k].append(v)
            self.n_history[k].append(count)

    def average(
            self,
            n:int = 0
    ):
        """
        Average latest n values or all values
        """
        assert n >= 0
        for k in self.val_history:
            v = np.array(self.val_history[k][-n:])
            nums = np.array(self.n_history[k][-n:])
            avg = np.sum(v * nums) / np.sum(nums)
            self.output[k] = avg
        self.ready = True