# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Dict, Optional
from ..base_model_builder import BaseModelBuilder
from ..builder import MLDOCK_BUILDER, TASKS_MANAGER


@TASKS_MANAGER.register_module()
class DefaultMLDOCKBuilder(BaseModelBuilder):
    """
    Default Machine Learning Docking model builder.
    """

    def __init__(
            self,
            cfg: Dict,
            train_cfg: Optional[dict] = None,
            test_cfg: Optional[dict] = None,
    ):
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def build_model(self):
        return MLDOCK_BUILDER.build(
            self.cfg,
            default_args = dict(train_cfg = self.train_cfg,
                                test_cfg = self.test_cfg)
        )