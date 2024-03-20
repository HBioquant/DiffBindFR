# Copyright (c) MDLDrugLib. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Union, Optional, List, Tuple, Dict, Any
from collections import OrderedDict
import copy
import logging
import os.path as osp

import torch
from torch.optim import Optimizer

import druglib
from .parallel import is_module_wrapper
from .checkpoint import load_checkpoint
from .dist_utils import get_dist_info
from .hooks import HOOKS, Hook
from .log_buffer import LogBuffer
from .priority import Priority, get_priority
from .utils import get_time_str
from druglib.utils import TORCH_VERSION, digit_version


class Brain(metaclass=ABCMeta):
    """
    The Brain for base Runner, a training helper for PyTorch
    Args:
        model (:obj:`torch.nn.Module`): The model to be run.
        optimizer (dict or :obj:`torch.optim.Optimizer`): It can be either an
            optimizer (in most cases) or a dict of optimizers (in models that
            requires more than one optimizer, e.g., GAN).
        work_dir (str, optional): The working directory to save checkpoints
            and logs. Defaults to None.
        logger (:obj:`logging.Logger`): Logger used during training.
             Defaults to None. (The default value is just for backward
             compatibility)
        meta (dict | None): A dict records some import information such as
            environment info and seed, which will be logged in logger hook.
            Defaults to None.
        max_epochs (int, optional): Total training epochs.
        max_iters (int, optional): Total training iterations.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: Optional[Union[Dict[str, Optimizer], Optimizer]] = None,
            work_dir: Optional[str] = None,
            logger: Optional[logging.Logger] = None,
            meta: Optional[dict] = None,
            max_epochs: Optional[int] = None,
            max_iters: Optional[int] = None,
    ):
        # check whether model has `model.train_step()` method
        if is_module_wrapper(model):
            _model = model.module
        else:
            _model = model
        assert hasattr(_model, "train_step"), "Model has no `model.train_step()` method.`"
        assert hasattr(_model, "val_step"), "Model has no `model.val_step()` method.`"
        self.model = model

        # check optimizer is dict or torch.optim.Optimizer
        if isinstance(optimizer, dict):
            for n, optim in optimizer.items():
                if not isinstance(optim, Optimizer):
                    raise TypeError(
                        f'optimizer must be a dict of torch.optim.Optimizers, '
                        f'but optimizer["{n}"] is a {type(optim)}')
        elif not isinstance(optimizer, Optimizer) and optimizer is not None:
            raise TypeError(
                f'optimizer must be a torch.optim.Optimizer object, dict, or None, '
                f'but got {type(optimizer)}')
        self.optimizer = optimizer

        # check work_dir is a string or NoneType
        if druglib.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            druglib.mkdir_or_exists(work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # check meta is a dict or NoneType
        if meta is not None and not isinstance(meta, dict):
            raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
        self.meta = meta

        # check logger is a logging.Logger
        if not isinstance(logger, logging.Logger):
            raise TypeError(f'logger must be a logging.Logger object, '
                            f'but got {type(logger)}')
        self.logger = logger

        if max_iters is not None and max_epochs is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')
        self._max_epochs = max_epochs
        self._max_iters = max_iters

        # get model named from the model class.__name__
        if hasattr(model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        # Built-in non-writable attributes
        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        self._hooks = [] # Higher priority to lower priority
        self.mode = None # train or val
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        # TODO: more flexible and elegant is needed.
        self.log_buffer = LogBuffer()

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job. (distributed training)"""
        return self._world_size

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks with higher priority to lower priority."""
        return self._hooks

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """
        Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        if isinstance(self.optimizer, dict):
            lr = dict()
            for n, optim in self.optimizer.items():
                lr[n] = [group['lr'] for group in optim.param_groups]
        elif isinstance(self.optimizer, Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.'
            )
        return lr

    def current_momentum(self) -> Union[List[float], Dict[str, List[float]]]:
        """
        Get current momentums.

        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        # define a momentum getting function
        def _get_momentum(
                optimizer,
        ) -> List[float]:
            momentum = []
            for group in optimizer.param_groups:
                if 'momentum' in group.keys():
                    momentum.append(group['momentum'])
                elif 'betas' in group.keys():
                    momentum.append(group['betas'])
                else:
                    # no use momentum
                    momentum.append(0)
            return momentum
        if isinstance(self.optimizer, dict):
            momentum = dict()
            for n, optim in self.optimizer.items():
                momentum[n] = _get_momentum(optim)
        elif isinstance(self.optimizer, Optimizer):
            momentum = _get_momentum(self.optimizer)
        else:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.'
            )
        return momentum

    def get_hook_info(self) -> str:
        """Get hook info from each Hook stage."""
        stage_hook_map = {
            stage: [] for stage in Hook.stages
        }
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name
            except ValueError:
                priority = hook.priority
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for strigger_stage in hook.get_triggered_stages():
                stage_hook_map[strigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n--------------------------------------------------------'
                stage_hook_infos.append(info)

        return '\n'.join(stage_hook_infos)

    def register_hook(
            self,
            hook: Hook,
            priority: Union[str, int, Priority] = 'NORMAL',
    ):
        """
        Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError(
                '"priority" is a reserved attribute for hooks.'
                'It cannot be set in advance.'
            )
        priority = get_priority(priority)
        hook.priority = priority
        inserted = False
        # insert the hook to a sorted list as priority order
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority > self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break

        if not inserted:
            self._hooks.insert(0, hook)

    def register_hook_from_cfg(
            self,
            hook_cfg: dict,
    ):
        """
        Register a hook from its cfg.

       Args:
           hook_cfg (dict): Hook config. It should have at least keys 'type'
             and 'priority' indicating its type and priority.

       Note:
           The specific hook class to register should not use 'type' and
           'priority' arguments during initialization.
        """
        hook_cfg = hook_cfg.copy()
        priority = hook_cfg.pop('priority', 'NORMAL')
        hook = druglib.build_from_cfg(hook_cfg, HOOKS)
        self.register_hook(hook, priority = priority)

    def call_hook(
            self,
            fn_name: str,
    ):
        """
        Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        assert isinstance(fn_name, str), "`fn_name` must be string."
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(
            self,
            filename: str,
            map_location: Any = "cpu",
            strict: bool = False,
            revise_keys: List[Tuple[str]] = [(r'^module.', '')],
    ) -> Union[dict, OrderedDict]:
        return load_checkpoint(
            model = self.model,
            filename = filename,
            map_location = map_location,
            strict = strict,
            logger = self.logger,
            revise_keys = revise_keys,
        )

    def _AdamW_debug(
            self,
            optim: Optimizer
    ):
        """
        See issue: https://github.com/pytorch/pytorch/issues/80809#issuecomment-1173481031
            for error 'AssertionError: If capturable=False, state_steps should not be CUDA tensors.'
        """
        if isinstance(optim, torch.optim.AdamW) and \
            digit_version(TORCH_VERSION) > digit_version('1.11.0'):
            optim.param_groups[0]['capturable'] = True

    def resume(
            self,
            checkpoint,
            resume_optimizer:bool = True,
            map_location = 'default',
    ):
        if map_location == 'default':
            if torch.cuda.is_available():
                cuda_device = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location = lambda storage, loc: storage.cuda(cuda_device)
                )
            else:
                checkpoint = self.load_checkpoint(
                    checkpoint,
                )
        else:
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location = map_location
            )

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']

        # Re-calculate the number of interesting when resuming
        # models with different number of GPUs
        if 'config' in checkpoint['meta']:
            cfg = druglib.Config.fromstring(
                checkpoint['meta']['config'], file_format='.py'
            )
            previous_gpu_ids = cfg.get('gpu_ids', [])
            if len(previous_gpu_ids) > 0 and \
                    len(previous_gpu_ids) != self.world_size:
                self._iter = int(self._iter * len(previous_gpu_ids) / self.world_size)
                self.logger.info('the iteration number is changed due to '
                                 'change of GPU number')
        # resume all meta information from checkpoint
        self.meta = checkpoint['meta']

        # resume optimizer from checkpoint if agree
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self._AdamW_debug(self.optimizer)
            elif isinstance(self.optimizer, dict):
                for k in checkpoint['optimizer'].keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k]
                    )
                    self._AdamW_debug(self.optimizer[k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        self.logger.info('resume epoch %d, iter %d', self._epoch, self._iter)

    # Brain structure assembly apartment
    def register_lr_hook(
            self,
            lr_config:Optional[Union[dict, Hook]],
    ):
        if lr_config is None:
            return
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            policy_type = lr_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of Lr updater.
            # Since this is not applicable for `
            # CosineAnnealingLrUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'LrUpdaterHook'
            lr_config['type'] = hook_type
            hook = druglib.build_from_cfg(lr_config, HOOKS)
        else:
            hook = lr_config
        self.register_hook(hook, priority = "VERY_HIGH")

    def register_momentum_hook(
            self,
            momentum_config:Optional[Union[dict, Hook]],
    ):
        if momentum_config is None:
            return
        if isinstance(momentum_config, dict):
            assert 'policy' in momentum_config
            policy_type = momentum_config.pop('policy')
            # If the type of policy is all in lower case, e.g., 'cyclic',
            # then its first letter will be capitalized, e.g., to be 'Cyclic'.
            # This is for the convenient usage of momentum updater.
            # Since this is not applicable for
            # `CosineAnnealingMomentumUpdater`,
            # the string will not be changed if it contains capital letters.
            if policy_type == policy_type.lower():
                policy_type = policy_type.title()
            hook_type = policy_type + 'MomentumUpdaterHook'
            momentum_config['type'] = hook_type
            hook = druglib.build_from_cfg(momentum_config, HOOKS)
        else:
            hook = momentum_config
        self.register_hook(hook, priority='HIGH')

    def register_optimizer_hook(
            self,
            optimizer_config:Optional[Union[dict, Hook]],
    ):
        if optimizer_config is None:
            return
        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('type', 'OptimizerHook')
            hook = druglib.build_from_cfg(optimizer_config, HOOKS)
        else:
            hook = optimizer_config
        self.register_hook(hook, priority='ABOVE_NORMAL')

    def register_checkpoint_hook(
            self,
            checkpoint_config:Optional[Union[dict, Hook]],
    ):
        if checkpoint_config is None:
            return
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = druglib.build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook, priority='NORMAL')

    def register_timer_hook(
            self,
            timer_config: Optional[Union[dict, Hook]],
    ):
        if timer_config is None:
            return
        if isinstance(timer_config, dict):
            timer_config_ = copy.deepcopy(timer_config)
            hook = druglib.build_from_cfg(timer_config_, HOOKS)
        else:
            hook = timer_config
        self.register_hook(hook, priority='LOW')

    def register_logger_hooks(
            self,
            log_config:Optional[dict],
    ):
        if log_config is None:
            return
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            hook = druglib.build_from_cfg(info, HOOKS, default_args=dict(interval=log_interval))
            self.register_hook(hook, priority='VERY_LOW')

    def register_custom_hooks(
            self,
            custom_config:Optional[Union[List[Union[dict, Hook]], dict, Hook]],
    ):
        if custom_config is None:
            return
        if not isinstance(custom_config, list):
            custom_config = [custom_config]
        for item in custom_config:
            if isinstance(item, dict):
                self.register_hook_from_cfg(item)
            else:
                self.register_hook(item, priority='NORMAL')

    def register_profiler_hook(
            self,
            profiler_config: Optional[Union[Hook, dict]],
    ):
        if profiler_config is None:
            return
        if isinstance(profiler_config, dict):
            profiler_config.setdefault('type', 'ProfilerHook')
            hook = druglib.build_from_cfg(profiler_config, HOOKS)
        else:
            hook = profiler_config
        self.register_hook(hook)

    def register_training_hooks(
            self,
            lr_config: Optional[Union[dict, Hook]],
            optimizer_config: Optional[Union[dict, Hook]] = None,
            checkpoint_config: Optional[Union[dict, Hook]] = None,
            log_config: Optional[dict] = None,
            momentum_config: Optional[Union[dict, Hook]] = None,
            timer_config: Optional[Union[dict, Hook]] = dict(type='IterTimerHook'),
            custom_hooks_config: Optional[Union[List[Union[dict, Hook]], dict, Hook]] = None,
    ):
        """
        Register default and custom hooks for training.

        Default and custom hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | MomentumUpdaterHook  | HIGH (30)               |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | CustomHook(s)        | defaults to NORMAL (50) |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        self.register_lr_hook(lr_config)
        self.register_momentum_hook(momentum_config)
        self.register_optimizer_hook(optimizer_config)
        self.register_checkpoint_hook(checkpoint_config)
        self.register_timer_hook(timer_config)
        self.register_logger_hooks(log_config)
        self.register_custom_hooks(custom_hooks_config)


class BaseRunner(Brain):
    """
    All subclass should implement the following APIs:
    -`run()`
    -`train()`
    -`val()`
    -`save_checkpoint()`
    """

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def val(self):
        pass

    @abstractmethod
    def run(
            self,
            data_loaders,
            workflow,
            **kwargs
    ):
        pass

    @abstractmethod
    def save_checkpoint(
            self,
            out_dir:str,
            filename_tmpl:str,
            save_optimizer:bool = True,
            meta:dict = None,
            create_symlink: bool = True
    ):
        pass

def clean_model_grad(
        model: torch.nn.Module,
):
    for p in model.parameters():
        if p.grad is not None:
            del p.grad
    torch.cuda.empty_cache()
