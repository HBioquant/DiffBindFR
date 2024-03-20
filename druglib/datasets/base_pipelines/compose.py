# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Callable, List, Union, Tuple
from collections.abc import Sequence, Mapping

from druglib.utils import build_from_cfg
from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """
    Basic function to compose multiple sequential transforms.
    `PyTorch Geometric` and `mmcv` extended libraries compatible.
    Args:
        transforms: Sequence[Mapping|callable]: a sequence of transform object or
            config mapping type to be composed.
    """
    def __init__(
            self,
            transforms: Union[List[Mapping], List[Callable]],
    ):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for t in transforms:
            if isinstance(t, Mapping):
                transform = build_from_cfg(t, PIPELINES)
                self.transforms.append(transform)
            elif isinstance(t, Callable):
                self.transforms.append(t)
            else:
                raise TypeError(f"The elements of transforms must be dict or callable, but got {type(t)}")

    def __call__(
            self,
            data: Union[List[Mapping], Tuple[Mapping], Mapping],
    ):
        if isinstance(data, (list, tuple)):
            return [self(d) for d in data]
        elif isinstance(data, Mapping):
            for t in self.transforms:
                data = t(data)
                if data is None:
                    return None
            return data
        return data

    def __repr__(self):
        string_formated = self.__class__.__name__ + "("
        for t in self.transforms:
            str_ = t.__repr__()
            if "Compose(" in str_:
                str_ = str_.replace("\n", "\n    ")
            string_formated += "\n"
            string_formated += f"    {str_}"
        string_formated += "\n)"
        return string_formated

