# Copyright (c) MDLDrugLib. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseModelBuilder(metaclass=ABCMeta):

    @abstractmethod
    def build_model(self):
        pass