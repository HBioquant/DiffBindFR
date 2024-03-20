# Copyright (c) MDLDrugLib. All rights reserved.
import copy
from typing import Any, Optional

from ...utils import Registry

RUNNERS = Registry("runner")
RUNNER_BUILDERS = Registry("runner builder")

def build_runner_builder(
        cfg:dict
) -> Any:
    return RUNNER_BUILDERS.build(cfg)

def build_runner(
        cfg:dict,
        default_args:Optional[dict] = None,
) -> Any:
    runner_cfg = cfg.copy()
    builder_type = runner_cfg.pop(
        "RunnerBuilder",
        "DefaultRunnerBuilder"
    )
    runner_builder = build_runner_builder(
        dict(
            type = builder_type,
            runner_cfg = runner_cfg,
            default_args = default_args,
        )
    )
    runner = runner_builder()
    return runner