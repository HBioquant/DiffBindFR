# Copyright (c) MDLDrugLib. All rights reserved.
from .builder import RUNNERS, build_runner
from .priority import get_priority, Priority
from .log_buffer import LogBuffer
from .default_RunnerBuilder import DefaultRunnerBuilder
from .utils import get_host_info, get_time_str, obj_from_dict, set_random_seed, multi_apply, unmap, init_random_seed
from .base_module import BaseModule, Sequential, ModuleDict, ModuleList
from .dist_utils import (allreduce_grads, allreduce_params, get_dist_info, master_only, setup_multi_processes,
                         init_dist, get_cpu_count, get_group, synchronize, init_process_group, all_reduce_dict,
                         reduce_mean, tensor2obj, obj2tensor, set_cuda_visible_device, get_available_gpu)
from .parallel import (MODULE_WRAPPERS, is_module_wrapper, scatter, scatter_kwargs,
                       MDLDataParallel, MDLDistributedDataParallel, collate, is_module_wrapper,
                       is_mlu_available, get_device, build_dp, build_ddp, )

from .fp16_utils import is_fp16_enabled, cast_tensor_type, auto_fp16, force_fp32, LossScaler, wrap_fp16_model
from .checkpoint import (CheckpointLoader, _load_checkpoint, _load_checkpoint_with_prefix, load_checkpoint,
                         load_state_dict, save_checkpoint, weights_to_cpu, find_latest_checkpoint)

from .hooks import (HOOKS, CheckpointHook, ClearMLLoggerHook, ClosureHook,
                    DistEvalHook, DistSamplerSeedHook, DvcliveLoggerHook,
                    EMAHook, EvalHook, Fp16OptimizerHook, ExpMomentumEMAHook,
                    GradientCumulativeFp16OptimizerHook, LinearMomentumEMAHook,
                    GradientCumulativeOptimizerHook, Hook, IterTimerHook,
                    LoggerHook, MlflowLoggerHook, NeptuneLoggerHook,
                    OptimizerHook, PaviLoggerHook, SegmindLoggerHook,
                    SyncBuffersHook, TensorboardLoggerHook, TextLoggerHook,
                    WandbLoggerHook)
from .hooks.lr_updater import StepLrUpdaterHook
from .hooks.lr_updater import (AnnealingLrUpdaterHook,CosineRestartLrUpdaterHook,
                               CyclicLrUpdaterHook, ExpLrUpdaterHook, FixedLrUpdaterHook,
                               FlatCosineAnnealingLrUpdaterHook, InvLrUpdaterHook, LrUpdaterHook,
                               OneCycleLrUpdaterHook, PolyLrUpdaterHook)
from .hooks.momentum_updater import (AnnealingMomentumUpdaterHook,
                                     CyclicMomentumUpdaterHook, MomentumUpdaterHook,
                                     OneCycleMomentumUpdaterHook, StepMomentumUpdaterHook)
from .base_runner import Brain, BaseRunner
from .epoch_based_runner import EpochBasedRunner
from .iter_based_runner import IterBasedRunner
from .optimizer import OPTIMIZERS, OPTIMIZERS_BUILDERS, builder_optimizer, DefaultOptimizerBuilder, Lion
from .engine import single_gpu_inference, multi_gpu_inference, default_argument_parser


__all__ = [
    'RUNNERS', 'build_runner', 'get_host_info', 'get_time_str', 'obj_from_dict', 'set_random_seed',
    'allreduce_grads', 'allreduce_params', 'get_dist_info', 'master_only', 'init_dist', 'get_cpu_count',
    'get_group', 'synchronize', 'init_process_group', 'DefaultRunnerBuilder', 'LogBuffer', 'MODULE_WRAPPERS',
    'is_module_wrapper', 'is_fp16_enabled', 'cast_tensor_type', 'auto_fp16', 'force_fp32', 'LossScaler', 'wrap_fp16_model',
    'CheckpointLoader', '_load_checkpoint', '_load_checkpoint_with_prefix', 'load_checkpoint', 'load_state_dict',
    'save_checkpoint', 'weights_to_cpu', 'get_priority', 'Priority', 'BaseModule', 'Sequential', 'ModuleDict',
    'ModuleList', 'Brain', 'BaseRunner', 'HOOKS', 'CheckpointHook', 'ClearMLLoggerHook', 'ClosureHook', 'DistEvalHook',
    'DistSamplerSeedHook', 'DvcliveLoggerHook', 'EMAHook', 'EvalHook', 'Fp16OptimizerHook', 'GradientCumulativeFp16OptimizerHook',
    'GradientCumulativeOptimizerHook', 'Hook', 'IterTimerHook', 'LoggerHook', 'MlflowLoggerHook', 'NeptuneLoggerHook',
    'OptimizerHook', 'PaviLoggerHook', 'SegmindLoggerHook', 'SyncBuffersHook', 'TensorboardLoggerHook',
    'TextLoggerHook', 'WandbLoggerHook', 'StepLrUpdaterHook', 'AnnealingLrUpdaterHook', 'CosineRestartLrUpdaterHook',
    'CyclicLrUpdaterHook', 'ExpLrUpdaterHook', 'FixedLrUpdaterHook', 'FlatCosineAnnealingLrUpdaterHook', 'InvLrUpdaterHook',
    'LrUpdaterHook', 'OneCycleLrUpdaterHook', 'PolyLrUpdaterHook', 'AnnealingMomentumUpdaterHook', 'CyclicMomentumUpdaterHook',
    'MomentumUpdaterHook', 'OneCycleMomentumUpdaterHook', 'StepMomentumUpdaterHook', 'EpochBasedRunner', 'IterBasedRunner',
    'OPTIMIZERS', 'builder_optimizer', 'OPTIMIZERS_BUILDERS', 'scatter', 'scatter_kwargs', 'MDLDataParallel',
    'MDLDistributedDataParallel', 'collate', 'is_module_wrapper', 'is_mlu_available', 'get_device', 'build_dp', 'build_ddp',
    'setup_multi_processes', 'find_latest_checkpoint', 'single_gpu_inference', 'multi_gpu_inference', 'default_argument_parser',
    'all_reduce_dict', 'reduce_mean', 'tensor2obj', 'obj2tensor', 'multi_apply', 'unmap', 'init_random_seed',
    'DefaultOptimizerBuilder', 'ExpMomentumEMAHook', 'LinearMomentumEMAHook', 'Lion', 'set_cuda_visible_device', 'get_available_gpu',

]
