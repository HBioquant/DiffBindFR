# Copyright (c) MDLDrugLib. All rights reserved.
from dataclasses import dataclass
from .feature_store import TensorAttr, _field_status


@dataclass
class CVTensorAttr(TensorAttr):
    """Attribute class for CV Data, whose `group_name` is 'cv'."""
    def __init__(
            self,
            attr_name = _field_status.UNSET,
            index = _field_status.UNSET,
    ):
        # Treat group_name as optional, and move it to the end
        super().__init__("cv", attr_name, index)
