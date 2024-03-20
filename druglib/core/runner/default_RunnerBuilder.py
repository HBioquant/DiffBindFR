# Copyright (c) MDLDrugLib. All rights reserved.
from .builder import RUNNERS, RUNNER_BUILDERS

@RUNNER_BUILDERS.register_module()
class DefaultRunnerBuilder:
    """
    Default builder for runners
    Custom existing `Runner` like `EpochBasedRunner` though `RunnerBuilder`.
    For example, we can inject some new properties and functions for `Runner`.
    E.g.
        >>> from druglib.core.runner.builder import RUNNER_BUILDERS, build_runner, RUNNERS
        >>> # Define a new RunnerRebuilder
        >>> @@RUNNER_BUILDERS.register_module()
        >>> class MyRunnerBuilder:
        ...     def __init__(self, runner_cfg, default_args = None):
        ...         if not isinstance(runner_cfg, dict):
        ...             raise TypeError('runner_cfg should be a dict',
        ...                             f'but got {type(runner_cfg)}')
        ...         self.runner_cfg = runner_cfg
        ...         self.default_args = default_args
        ...
        ...     def __call__(self):
        ...         runner = RUNNERS.build(self.runner_cfg,
        ...                                default_args=self.default_args)
        ...         # Add new properties for existing runner
        ...         runner.my_name = 'my_runner'
        ...         runner.my_function = lambda self: print(self.my_name)
        ...         ...
        >>> # build your runner
        >>> runner_cfg = dict(type='EpochBasedRunner', max_epochs=40,
        ...                   constructor='MyRunnerConstructor')
        >>> runnerbuilder_cfg = dict(type='MyRunnerConstructor', runner_cfg=runner_cfg, default_args=None)
        >>> runner = build_runner(runnerbuilder_cfg)
    """
    def __init__(
            self,
            runner_cfg:dict,
            default_args = None,
    ):
        if not isinstance(runner_cfg, dict):
            raise TypeError(
                f'`runner_cfg` must be a dict, but got {type(runner_cfg)}.'
            )
        self.runner_cfg = runner_cfg
        self.default_args = default_args

    def __call__(self):
        return RUNNERS.build(
            self.runner_cfg,
            default_args = self.default_args
        )