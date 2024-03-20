# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, Union, List, Tuple, Dict, Callable

from druglib import is_list_of
from .hook import HOOKS, Hook
from .lr_updater import format_param, annealing_cos, annealing_linear


class MomentumUpdaterHook(Hook):
    """
    Momentum Scheduler.
    Model -> base_momentum ---> self.get_regular_momentum (dependent on specific `get_momentum` function)
    -> regular_momentum ---> self._set_momentum -> momentum
    Args:
        by_epoch (bool): momentum changes epoch by epoch
            Default: True
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
            Default: None
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
            Default: 0
        warmup_ratio (float): momentum used at the beginning of warmup equals to
            warmup_ratio * initial_momentum
            Default: 0.1
    """
    def __init__(
            self,
            by_epoch: bool = True,
            warmup: Optional[str] = None,
            warmup_iters: int = 0,
            warmup_ratio: float = 0.9,
    ):
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant", "exp" and "linear"')
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup =  warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

        self.base_momentum: Union[list, dict] = []  # initial momentum for all param groups
        self.regular_momentum: Union[list, dict] = []  # expected momentum if no warming up is performed

    def _set_momentum(
            self,
            runner,
            momentum_groups,
    ):
        """set optimizer learning rate"""
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, momentum in zip(optim.param_groups, momentum_groups[k]):
                    if 'momentum' in param_group.keys():
                        param_group['momentum'] = momentum
                    elif 'betas' in param_group.keys():
                        param_group['betas'] = (momentum, param_group['betas'][1])
        else:
            for param_group, momentum in zip(runner.optimizer.param_groups, momentum_groups):
                if 'momentum' in param_group.keys():
                    param_group['momentum'] = momentum
                elif 'betas' in param_group.keys():
                    param_group['betas'] = (momentum, param_group['betas'][1])

    def get_warmup_momentum(
            self,
            cur_iters: int,
    ) -> Union[Dict[str, list], list]:
        """get transformed learning rate according to warmup iterations"""
        def _get_warmup_momentum(
                cur_iters: int,
                regular_momentum: list,
        ) -> list:
            if self.warmup == "constant":
                warmup_momentum = [_momentum * self.warmup_ratio for _momentum in regular_momentum ]
            elif self.warmup == "linear":
                # define a linear decay coefficient
                k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
                warmup_momentum = [_momentum * (1 - k) for _momentum in regular_momentum]
            elif self.warmup == "exp":
                # define a exponential warmup coefficient
                k = self.warmup_ratio ** (1 - cur_iters / self.warmup_iters)
                warmup_momentum = [_momentum * k for _momentum in regular_momentum]
            else:
                raise NotImplementedError(f'"{self.warmup}" is not a supported type for warming up, valid'
                    ' types are "constant", "linear" and "exp"')

            return warmup_momentum

        if isinstance(self.regular_momentum, dict):
            momentum_groups = {}
            for key, regular_momentum in self.regular_momentum.items():
                momentum_groups[key] = _get_warmup_momentum(cur_iters, regular_momentum)
            return momentum_groups
        else:
            return _get_warmup_momentum(cur_iters, self.regular_momentum)

    # custom momentum updater hook
    def get_momentum(
            self,
            runner,
            base_momentum,
    ) -> float:
        raise NotImplementedError

    def get_regular_momentum(
            self,
            runner,
    ) -> Union[Dict[str, list], list]:
        if isinstance(runner.optimizer, dict):
            assert isinstance(self.base_momentum, dict)
            momentum_groups: Dict[str, List[float]] = dict()
            for k in runner.optimizer.keys():
                momentum_groups.update({k: [self.get_momentum(runner, _momentum) for _momentum in self.base_momentum[k]]})
            return momentum_groups
        else:
            assert isinstance(self.base_momentum, list)
            return [self.get_momentum(runner, _momentum) for _momentum in self.base_momentum]

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_momentum' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_momentum = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    # if group has no `initial_momentum`, then give optimizer's `momentum` to group's `initial_momentum`
                    if "momentum" in group.keys():
                        group.setdefault('initial_momentum', group['momentum'])
                    else:
                        group.setdefault('initial_momentum', group['betas'][0])
                _base_momentum = [
                    group['initial_momentum'] for group in optim.param_groups
                ]
                self.base_momentum.update({k: _base_momentum})
        else:
            for group in runner.optimizer.param_groups:
                if "momentum" in group.keys():
                    group.setdefault('initial_momentum', group['momentum'])
                else:
                    group.setdefault('initial_momentum', group['betas'][0])
            self.base_momentum = [
                group['initial_momentum'] for group in runner.optimizer.param_groups
            ]

    def before_train_epoch(self, runner):
        # if by iteractions, this method `before_train_epoch` do nothing
        if not self.by_epoch:
            return
        # epoch by epoch: base momentum -> regular momentum -> set momentum
        self.regular_momentum = self.get_regular_momentum(runner)
        self._set_momentum(runner, self.regular_momentum)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_momentum(runner, self.regular_momentum)
            else:
                warmup_momentum = self.get_warmup_momentum(cur_iter)
                self._set_momentum(runner, warmup_momentum)
        else:
            self.regular_momentum = self.get_regular_momentum(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_momentum(runner, self.regular_momentum)
            else:
                warmup_momentum = self.get_warmup_momentum(cur_iter)
                self._set_momentum(runner, warmup_momentum)

@HOOKS.register_module()
class StepMomentumUpdaterHook(MomentumUpdaterHook):
    """
    Step momentum scheduler with min_momentum clipping.
    Args:
        step (int | list[int]): Step to decay the momentum. If an int value is given,
            regard it as the decay interval. If a list is given, decay momentum at
            these steps.
        gamma (float, optional): Decay momentum ratio. Default: 0.5.
        min_momentum (float, optional): Minimum momentum value to keep. If momentum after decay
            is lower than `min_momentum`, it will be clipped to this value. If None
            is given, we don't perform momentum clipping. Default: None.
    """

    def __init__(
            self,
            step: Union[int,list],
            gamma: float = 0.5,
            min_momentum: Optional[float] = None,
            **kwargs
    ):
        if isinstance(step, list):
            assert is_list_of(step, int)
            assert all([s > 0 for s in step])
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError("`step` must be a list of int or int.")
        self.step = step
        self.gamam = gamma
        self.min_momentum = min_momentum

        super(StepMomentumUpdaterHook, self).__init__(**kwargs)

    def get_momentum(
            self,
            runner,
            base_momentum,
    ):
        progress = runner.epoch if self.by_epoch else runner.iter

        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        momentum = base_momentum * (self.gamma ** exp)
        if self.min_momentum is not None:
            # clip to a minimum value
            momentum = max(momentum, self.min_momentum)
        return momentum

@HOOKS.register_module()
class AnnealingMomentumUpdaterHook(MomentumUpdaterHook):
    """
    Cosine (or linear) annealing LR Momentum decays the Momentum of each parameter group linearly.

    Args:
        min_momentum (float, optional): The minimum momentum. Default: None.
        min_momentum_ratio (float, optional): The ratio of minimum momentum to the base momentum.
            Either `min_momentum` or `min_momentum_ratio` should be specified.
            Default: None.
        annealing_type (str, optional): 'cos' or 'linear'. Default: 'cos'.
    """
    annealing_fn = {
        'cos': annealing_cos,
        'linear': annealing_linear,
    }
    def __init__(
            self,
            min_momentum: Optional[float] = None,
            min_momentum_ratio: Optional[float] = None,
            annealing_type: str = 'cos',
            **kwargs
    ):
        assert isinstance(annealing_type, str), f"`annealing_type` must be string, but got {type(annealing_type)}"
        if annealing_type.lower() not in ['cos', 'linear']:
            raise ValueError(f'`annealing_type` must be either `cos` or `linear`, but got {annealing_type}')
        self.annealing_fn =  self.annealing_fn[annealing_type]
        assert (min_momentum is None) ^ (min_momentum_ratio is None)
        self.min_momentum = min_momentum
        self.min_momentum_ratio = min_momentum_ratio
        super(AnnealingMomentumUpdaterHook, self).__init__(**kwargs)

    def get_momentum(
            self,
            runner,
            base_momentum,
    ):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        if self.min_momentum_ratio is not None:
            target_momentum = base_momentum * self.min_momentum_ratio
        else:
            target_momentum = self.min_momentum
        return self.annealing_fn(
            start = base_momentum,
            end = target_momentum,
            factor = progress / max_progress,
        )

@HOOKS.register_module()
class CyclicMomentumUpdaterHook(MomentumUpdaterHook):
    """
    Cyclic momentum Scheduler.

    Implement the cyclical momentum scheduler policy described in
    https://arxiv.org/pdf/1708.07120.pdf

    This momentum scheduler usually used together with the CyclicLRUpdater
    to improve the performance in the 3D detection area.

    Args:
        by_epoch (bool, optional): Whether to update momentum by epoch. False
        target_ratio (tuple[float], optional): Relative ratio of the highest momentum
            and the lowest momentum to the initial momentum. (High, Low) or float -> (x, x/1e5)
            If you input float such as 1e-9, we suggest you reset the `target_ratio` value.
        cyclic_times (int, optional): Number of cycles during training
        step_ratio_up (float, optional): The ratio of the increasing process of
            momentum in the total cycle.
        anneal_strategy (str, optional): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing. Default: 'cos'.
        gamma (float, optional): Cycle decay ratio. Default: 1.
            It takes values in the range (0, 1]. The difference between the
            maximum learning rate and the minimum learning rate decreases
            periodically when it is less than 1.
    """
    annealing_fn = {
        "cos": annealing_cos,
        "linear": annealing_linear,
    }

    def __init__(
            self,
            by_epoch: bool = False,
            target_ratio: Tuple[float, float] = (0.85 / 0.95, 1.),
            cyclic_times: int = 1,
            step_ratio_up: float = 0.4,
            anneal_strategy: str = 'cos',
            gamma: Union[float] = 1.,
            **kwargs
    ):
        if isinstance(target_ratio, float):
            target_ratio = (target_ratio, target_ratio/1e5)
        elif isinstance(target_ratio, tuple):
            target_ratio = (target_ratio[0], target_ratio[0]/1e5) if len(target_ratio) == 1 else target_ratio
        else:
            raise ValueError(
                f"`target_ratio` should be either float or tuple, got {type(target_ratio)}"
            )
        assert len(target_ratio) == 2, \
            '"target_ratio" must be list or tuple of two floats'
        assert 0 <= step_ratio_up < 1.0, \
            '"step_ratio_up" must be in range [0,1)'
        assert 0 < gamma <= 1, \
            '"gamma" must be in range (0, 1]'

        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.gamma = gamma
        self.max_iter_per_phase = None
        self.momentum_phases: List[list] = []  # init momentum_phases
        # validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError('anneal_strategy must be one of "cos" or '
                             f'"linear", instead got {anneal_strategy}')
        self.anneal_func = self.annealing_fn[anneal_strategy]

        assert not by_epoch, \
            'currently only support "by_epoch" = False in `CyclicMomentumUpdaterHook`'
        super(CyclicMomentumUpdaterHook, self).__init__(by_epoch, **kwargs)

    def before_run(self, runner):
        super(CyclicMomentumUpdaterHook, self).before_run(runner)
        # initiate momentum_phases
        # total momentum_phases are separated as up and down
        self.max_iter_per_phase = runner.max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * self.max_iter_per_phase)
        # add up phase: 0 iter -> iter_up_phase iter; base momentum -> base_momentum * up_ratio
        self.momentum_phases.append([0, iter_up_phase, 1, self.target_ratio[0]])
        # add down phase: iter_up_phase iter -> max iter per phase; base momentum * up_ratio -> base momentum * down_ratio
        self.momentum_phases.append([iter_up_phase, self.max_iter_per_phase, self.target_ratio[0], self.target_ratio[1]])

    def get_momentum(
            self,
            runner,
            base_momentum,
    ):
        curr_cycle = runner.iter // self.max_iter_per_phase
        curr_iter = runner.iter % self.max_iter_per_phase
        # update weight decay
        k = self.gamma**curr_cycle

        for (start_iter, end_iter, start_ratio, end_ratio) in self.momentum_phases:
            if start_iter <= curr_iter < end_iter:
                # Apply cycle scaling to gradually reduce the difference
                # between max_momentum and base momentum. The target end_ratio can be
                # expressed as:
                # end_ratio = (base_momentum + k * (max_momentum - base_momentum)) / base_momentum
                # iteration: 0 --> iter_up_phase:
                if start_iter == 0:
                    end_ratio = 1 - k + k * end_ratio
                else:
                    # iteration: iter_up_phase --> max_iter_per_phase:
                    start_ratio = 1 - k + k * start_ratio
                progress = curr_iter - start_iter
                return self.anneal_func(
                    base_momentum * start_ratio,
                    base_momentum * end_ratio,
                    progress / (end_iter - start_iter),
                )
        raise RuntimeError('The method should return in the for-loop and '
                           'should not be executed until this')

@HOOKS.register_module()
class OneCycleMomentumUpdaterHook(MomentumUpdaterHook):
    """
    One Cycle momentum Scheduler.

    This momentum scheduler usually used together with the OneCycleLrUpdater
    to improve the performance.

    Args:
        max_momentum (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is
            'max_momentum' and learning rate is 'base_lr'
            Default: 0.95
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
        step_ratio_up (float): The percentage of the cycle (in number of steps)
            spent increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing.
            Default: 'cos'
        three_phase (bool): If three_phase is True, use a third phase of the
            schedule to annihilate the learning rate according to
            final_div_factor instead of modifying the second phase (the first
            two phases will be symmetrical about the step indicated by
            pct_start).
            Default: False
    """
    annealing_fn = {
        "cos": annealing_cos,
        "linear": annealing_linear,
    }

    def __init__(
            self,
            base_momentum: Union[float, list, dict] = 0.85,
            max_momentum: Union[float, list, dict] = 0.95,
            step_ratio_up: float = 0.3,
            anneal_strategy: str = 'cos',
            three_phase: bool = False,
            **kwargs
    ):
        # validate by_epoch, currently only support by_epoch = False
        if 'by_epoch' not in kwargs:
            kwargs['by_epoch'] = False
        else:
            assert not kwargs['by_epoch'], \
                'currently only support "by_epoch" = False'
        if not isinstance(base_momentum, (float, list, dict)):
            raise ValueError('base_momentum must be the type among of float,'
                             'list or dict.')
        self._base_momentum = base_momentum
        if not isinstance(max_momentum, (float, list, dict)):
            raise ValueError('the type of max_momentum must be the one of list or '
                             f'dict, but got {type(max_momentum)}')
        self._max_momentum = max_momentum
        # validate pct_start
        if step_ratio_up < 0 or step_ratio_up > 1 or not isinstance(step_ratio_up, float):
            raise ValueError('expected float between 0 and 1 pct_start, but '
                             f'got {step_ratio_up}')
        self.step_ratio_up = step_ratio_up
        # validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError('anneal_strategy must be one of "cos" or '
                             f'"linear", instead got {anneal_strategy}')
        self.anneal_func = self.annealing_fn[anneal_strategy]
        self.three_phase = three_phase
        self.momentum_phases: list = []  # init momentum_phases
        super(OneCycleMomentumUpdaterHook, self).__init__(**kwargs)

    def before_run(self, runner):
        if isinstance(runner.optimizer, dict):
            self._base_momentum = {}
            for k, optim in runner.optimizer.items():
                if 'momentum' not in optim.defaults and 'betas' not in optim.defaults:
                    raise ValueError('optimizer must support momentum with'
                                     'option enabled')
                self.use_beta1 = 'betas' in optim.defaults
                _base_momentum = format_param(k, optim, self._base_momentum)
                _max_momentum = format_param(k, optim, self._max_momentum)
                for group, b_momentum, m_momentum in zip(
                        optim.param_groups, _base_momentum, _max_momentum):
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (m_momentum, beta2)
                    else:
                        group['momentum'] = m_momentum
                    group['base_momentum'] = b_momentum
                    group['max_momentum'] = m_momentum
        else:
            optim = runner.optimizer
            if 'momentum' not in optim.defaults and 'betas' not in optim.defaults:
                raise ValueError('optimizer must support momentum with'
                                 'option enabled')
            self.use_beta1 = 'betas' in optim.defaults
            k = type(optim).__name__
            _base_momentum = format_param(k, optim, self._base_momentum)
            _max_momentum = format_param(k, optim, self._max_momentum)
            self._base_momentum = [momentum / self.div_factor for momentum in _max_momentum]
            for group, b_momentum, m_momentum in zip(
                    optim.param_groups, _base_momentum, _max_momentum):
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (m_momentum, beta2)
                else:
                    group['momentum'] = m_momentum
                group['base_momentum'] = b_momentum
                group['max_momentum'] = m_momentum

        total_steps = runner.max_iters
        if self.three_phase:
            self.momentum_phases.append(
                [float(self.step_ratio_up * total_steps) - 1, self._max_momentum, self._base_momentum])
            self.momentum_phases.append(
                [float(2 * self.step_ratio_up * total_steps) - 2, self._base_momentum, self._max_momentum])
            self.momentum_phases.append(
                [total_steps - 1, self._max_momentum, self._max_momentum])
        else:
            self.momentum_phases.append(
                [float(self.step_ratio_up * total_steps) - 1, self._max_momentum, self._base_momentum])
            self.momentum_phases.append(
                [total_steps - 1, self._base_momentum, 1 / self._max_momentum])

    def get_momentum(
            self,
            runner,
            base_momentum,
    ):
        curr_iter = runner.iter
        start_iter = 0
        momentum = 0.
        for i, (end_iter, start_momentum, end_momentum) in enumerate(self.momentum_phases):
            if curr_iter <= end_iter or i == (len(self.momentum_phases) - 1):
                pct = (curr_iter - start_iter) / (end_iter - start_iter)
                momentum = self.anneal_func(start_momentum, end_momentum, pct)
                break
            start_iter = end_iter
        return momentum
