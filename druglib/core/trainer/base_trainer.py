# Copyright (c) MDLDrugLib. All rights reserved.
import os, logging
from typing import List, Union, Optional

import torch.nn as nn
from torch.utils.data import Dataset

from druglib.utils import (Config, compat_cfg,)
from ..runner import (EpochBasedRunner, OptimizerHook, Fp16OptimizerHook, find_latest_checkpoint,
                      DistSamplerSeedHook, builder_optimizer, build_runner,EvalHook, DistEvalHook,
                      build_ddp, build_dp, get_dist_info, )
from druglib.datasets import build_dataset, build_dataloader


def auto_scale_lr(
        cfg: Config,
        distributed: bool,
        logger: logging.Logger
):
    """
    Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg: config, training config.
        distributed: bool, using distributed or not.
        logger: logging.Logger, Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')

def train_model(
        model: nn.Module,
        dataset: Union[Dataset, List[Dataset]],
        cfg: Config,
        logger:logging.Logger,
        distributed: bool = False,
        validate: bool = False,
        timestamp: Optional[str] = None,
        meta: Optional[dict] = None,
):
    cfg = compat_cfg(cfg)

    # 1. prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # 2. put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            fp16_compression = cfg.get('fp16_compression', False),
            device_ids = [int(os.environ['LOCAL_RANK'])],
            broadcast_buffers = False,
            find_unused_parameters = find_unused_parameters,
        )
    else:
        model = build_dp(
            model,
            cfg.device,
            device_ids = cfg.gpu_ids,
        )

    # 3. build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = builder_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args = dict(
            model = model,
            optimizer = optimizer,
            work_dir = cfg.work_dir,
            logger = logger,
            meta = meta,
            enable_avoid_omm = False))

    #  make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config = cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False,
        )

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        val_dataset = build_dataset(cfg.data.val)
        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # Note that the priority of IterTimerHook is 'LOW'.
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority = 'LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)