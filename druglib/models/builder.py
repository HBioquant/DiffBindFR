# Copyright (c) MDLDrugLib. All rights reserved.
import copy
from typing import Optional
from .base_model_builder import BaseModelBuilder
from druglib.utils import Registry, build_from_cfg, Config

TASKS_MANAGER = Registry("tasks manager")

# task-wised model container
MLDOCK_BUILDER = Registry("mldock model builder")

# subtask module container
ENCODER = Registry("encoder")
ATTENTION = Registry("attention")
DIFFUSION = Registry("diffusion")
INTERACTION = Registry("interaction")
ENERGY = Registry("energy")


def build_encoder(cfg):
    """Build encoder block"""
    return ENCODER.build(cfg)

def build_attention(cfg):
    """Build attention block"""
    return ATTENTION.build(cfg)

def build_diffusion(cfg):
    """Build diffusion model"""
    return DIFFUSION.build(cfg)

def build_interaction(cfg):
    """Build interaction block"""
    return INTERACTION.build(cfg)

def build_energy(cfg):
    """Build energy block"""
    return ENERGY.build(cfg)

def build_task_builder(
        cfg: Config,
        default_args: Optional[dict] = None,
) -> BaseModelBuilder:

    return build_from_cfg(
        cfg,
        TASKS_MANAGER,
        default_args,
    )


def build_task_model(
        cfg: Config,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
):
    assert cfg.get("train_cfg") is None or train_cfg is None, "train_cfg specified in either outer field or model field"
    assert cfg.get("test_cfg") is None or test_cfg is None, "test_cfg specified in either outer field or model field"

    task_cfg = copy.deepcopy(cfg)
    task = task_cfg.pop(
        "task"
    )
    # upper string required for calling `task builder`
    task = task.upper()
    task_builder = task_cfg.pop(
        f"{task}Builder",
        f"Default{task}Builder",
    )
    builder_cfg = {
        "type": task_builder,
        "cfg": task_cfg
    }
    model_builder: BaseModelBuilder = build_task_builder(
        builder_cfg,
        default_args = dict(train_cfg = train_cfg, test_cfg = test_cfg)
    )

    return model_builder.build_model()