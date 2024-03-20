# Copyright (c) MDLDrugLib. All rights reserved.
import copy, warnings
from abc import ABCMeta
from collections import defaultdict
from logging import FileHandler

from torch import nn

from .dist_utils import master_only
from druglib.apis import initialize, update_init_info
from druglib.utils.logger import get_logger, logger_initialized, print_log


class BaseModule(nn.Module, metaclass=ABCMeta):
    """
    Base module for all modules in druglib.
    `BaseModule` is a wrapper of `torch.nn.Module` with additional
        functionality of parameter initialization. Compared with `torch.nn.Module`,
        `BaseModule` mainly adds three attributes.
    - `init_cfg`: The config to control the initialization.
    - `init_weights`: The function of parameter initialization and recording
        initialization information.
    - `_params_init_info`: Used to track the parameter initialization.
        information. This attribute only exists during executing the `init_weights`.
    Args:
        init_cfg:dict: Initialization config dict.
            Note that init_cfg can be defined in diferent levels, but init_cfg
            in low levels has a higher priority.
    """
    def __init__(self, init_cfg = None):
        super(BaseModule, self).__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)
        self.active_report = self.init_cfg.pop('active_report', False)

    @property
    def is_init(self) -> bool:
        return self._is_init

    def init_weights(self):
        """
        Initialize the weights
        """
        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # the `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info:str: The string that describes the initialization.
            # - tmp_mean_value:FloatTensor: The mean of the parameter,
            #       which indicates whether the parameter has ben modified.
            #   this attribute would be deleted after all parameters is initialized.
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # when detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param]['init_info'] = \
                f'The value is the same before and after calling `init_weights` '
                f'of {self.__class__.__name__} '
                self._params_init_info[param]['tmp_mean_value'] = \
                param.data.mean()

            # pass `_params_init_info` to all submodules
            # All submodules share the same `_params_init_info`,
            # so it will be updated when parameters are modified
            # at any level of the model.
            for submodule in self.modules():
                submodule._params_init_info = self._params_init_info

        # Get the initialized logger, if not exists,
        # create a logger named `druglib`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'druglib'

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    msg = f'Initialize module `{module_name}` with init_cfg {self.init_cfg}',
                    logger = logger_name
                )
                initialize(module = self, init_cfg = self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # Prevent the parameters of the pretrained model
                    # from being overwritten by the `init_weights`
                    if self.init_cfg['type'] == 'Pretrained':
                        return
            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        module = m,
                        init_info = f'Initialized by user-defined `init_weights` in {m.__class__.__name__}'
                    )
                elif isinstance(m, (nn.ModuleList, nn.Sequential)):
                    for layer in m:
                        if hasattr(m, 'init_weights'):
                            layer.init_weights()
                            update_init_info(
                                module = m,
                                init_info = f'Initialized by user-defined `init_weights` '
                                            f'in Sequence Module {m.__class__.__name__}'
                            )
            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has been called more than once')

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    @master_only
    def _dump_init_info(
            self,
            logger_name:str,
    ):
        """
        Dump the initialization information to a file named
            `initialization.log.json` in workdir.
        Args:
            logger_name:str: The name of logger.
        """
        logger = get_logger(name = logger_name)
        with_file_handler = False
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write(
                    'Name of parameter - Initialization information\n'
                )
                for name, param in self.named_parameters():
                    shape = self._shape(param)
                    handler.stream.write(
                        f'\n{name} - {shape}: '
                        f"\n{self._params_init_info[param]['init_info']} \n"
                    )
                with_file_handler = True
        if not with_file_handler and self.active_report:
            for name, param in self.named_parameters():
                shape = self._shape(param)
                print_log(
                    msg = f'\n{name} - {shape}: '
                          f"\n{self._params_init_info[param]['init_info']} \n",
                    logger = logger
                )

    def _shape(self, param):
        if isinstance(param, nn.parameter.UninitializedParameter):
            shape = (-1, '...')
        else:
            shape = param.shape

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s



class Sequential(BaseModule, nn.Sequential):

    def __init__(self, *args, init_cfg = None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)

class ModuleList(BaseModule, nn.ModuleList):

    def __init__(self, modules = None, init_cfg = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)

class ModuleDict(BaseModule, nn.ModuleDict):

    def __init__(self, modules = None, init_cfg = None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)
