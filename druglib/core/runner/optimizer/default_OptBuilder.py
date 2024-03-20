# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Union, Optional, List, Tuple, Dict
import torch, warnings
import torch.nn as nn
from torch.nn import GroupNorm, LayerNorm

from druglib.utils import _BatchNorm, _InstanceNorm, build_from_cfg, is_list_of
from .builder import OPTIMIZERS_BUILDERS, OPTIMIZERS


@OPTIMIZERS_BUILDERS.register_module()
class DefaultOptimizerBuilder:
    """
    Default constructor for optimizers.

    By default each parameter share the same optimizer settings, and we
    provide an argument ``paramwise_cfg`` to specify parameter-wise settings.
    It is a dict and may contain the following fields:

    - ``custom_keys`` (dict): Specified parameters-wise settings by keys. If
      one of the keys in ``custom_keys`` is a substring of the name of one
      parameter, then the setting of the parameter will be specified by
      ``custom_keys[key]`` and other setting like ``bias_lr_mult`` etc. will
      be ignored. It should be noted that the aforementioned ``key`` is the
      longest key that is a substring of the name of the parameter. If there
      are multiple matched keys with the same length, then the key with lower
      alphabet order will be chosen.
      ``custom_keys[key]`` should be a dict and may contain fields ``lr_mult``
      and ``decay_mult``. See Example 2 below.
    - ``bias_lr_mult`` (float): It will be multiplied to the learning
      rate for all bias parameters (except for those in normalization
      layers and offset layers of DCN).
    - ``bias_decay_mult`` (float): It will be multiplied to the weight
      decay for all bias parameters (except for those in
      normalization layers, depthwise conv layers, offset layers of DCN).
    - ``norm_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of normalization
      layers.
    - ``dwconv_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of depthwise conv
      layers.
    - ``dcn_offset_lr_mult`` (float): It will be multiplied to the learning
      rate for parameters of offset layer in the deformable convs
      of a model.
    - ``bypass_duplicate`` (bool): If true, the duplicate parameters
      would not be added into optimizer. Default: False.

    Note:

        1. If the option ``dcn_offset_lr_mult`` is used, the constructor will
        override the effect of ``bias_lr_mult`` in the bias of offset layer.
        So be careful when using both ``bias_lr_mult`` and
        ``dcn_offset_lr_mult``. If you wish to apply both of them to the offset
        layer in deformable convs, set ``dcn_offset_lr_mult`` to the original
        ``dcn_offset_lr_mult`` * ``bias_lr_mult``.

        2. If the option ``dcn_offset_lr_mult`` is used, the constructor will
        apply it to all the DCN layers in the model. So be careful when the
        model contains multiple DCN layers in places other than backbone.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are

                - `type`: class name of the optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, etc.
        paramwise_cfg (dict, optional): Parameter-wise options.

    Example 1:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> paramwise_cfg = dict(norm_decay_mult=0.)
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)

    Example 2:
        >>> # assume model have attribute model.backbone and model.cls_head
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, weight_decay=0.95)
        >>> paramwise_cfg = dict(custom_keys={
                'backbone': dict(lr_mult=0.1, decay_mult=0.9)})
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)
        >>> # Then the `lr` and `weight_decay` for model.backbone is
        >>> # (0.01 * 0.1, 0.95 * 0.9). `lr` and `weight_decay` for
        >>> # model.cls_head is (0.01, 0.95).
    """
    def __init__(
            self,
            optimizer_cfg: Dict,
            paramwise_cfg: Optional[Dict] = None,
    ):
        if not isinstance(optimizer_cfg, dict):
            raise TypeError(f'optimizer_cfg should be a dict, but got {type(optimizer_cfg)}')
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = paramwise_cfg if paramwise_cfg is not None else {}
        self.base_lr = self.optimizer_cfg.get("lr", None)
        self.base_weightdecay = self.optimizer_cfg.get('weight_decay', None)
        self._validate()

    def _validate(self):
        """Not every optimizer inputs weight decay."""
        if not isinstance(self.paramwise_cfg, dict):
            raise TypeError(f'paramwise_cfg should be None or a dict, but got {type(self.paramwise_cfg)}')

        if "custom_keys" in self.paramwise_cfg:
            if not isinstance(self.paramwise_cfg['custom_keys'], dict):
                raise TypeError(
                    'If specified, custom_keys must be a dict, '
                    f'but got {type(self.paramwise_cfg["custom_keys"])}')

            if self.base_weightdecay is None:
                for k, v in self.paramwise_cfg["custom_keys"].items():
                    if "decay_mult" in v:
                        raise ValueError('`weight_decay` should not be None in `optimizer_cfg` or lost')

        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in self.paramwise_cfg
                or 'norm_decay_mult' in self.paramwise_cfg
                or 'dwconv_decay_mult' in self.paramwise_cfg):
            if self.base_wd is None:
                raise ValueError('base_wd should not be None')

    def __call__(
            self,
            model: nn.Module,
    ):
        if hasattr(model, "module"):
            model:nn.Module = model.module

        if not self.paramwise_cfg:
            self.optimizer_cfg["params"] = model.parameters()
            return build_from_cfg(self.optimizer_cfg, OPTIMIZERS)

        params: List[Dict] = []
        # place_holder
        self._get_paramwise_args()
        self.adjust_lr_dw(
            params,
            model,
        )
        self.optimizer_cfg["params"] = params
        return build_from_cfg(self.optimizer_cfg, OPTIMIZERS)

    def _get_paramwise_args(self):
        self.custom_keys = self.paramwise_cfg.get("", {})
        # first sort with alphabet order and then sort with reversed len of str
        # such as ['rtsasdsa', 'asdf', 'sadf', 'sdas', 'asd', 'df']
        self.custom_keys = sorted(sorted(self.custom_keys), key = len, reverse = True)

        self.bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        self.bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        self.norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        self.dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)
        self.bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)
        self.dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', 1.)

    def _is_in(
            self,
            param_group:Dict,
            params: List[Dict],
    ) -> bool:
        assert is_list_of(params, dict)
        param_set  = set(param_group['params'])
        params_set = set()
        for p in params:
            params_set.update(set(p["params"]))
        return not param_set.isdisjoint(params_set)

    def adjust_lr_dw(
            self,
            params: List[Dict],
            module: nn.Module,
            prefix: str = '',
            is_dcn_module: bool = False, # TODO: see the below
    ):
        """
        Adjust lr and weight decay for some modules,
        and then add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups `param`, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (bool): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to False.
        """
        # whether is normalization layer, for weight decay
        is_norm = isinstance(module, (GroupNorm, LayerNorm,
                                      _BatchNorm, _InstanceNorm))
        # check is depth-wise conv, for weight decay
        is_dwconv = (isinstance(module, torch.nn.Conv2d) and module.in_channels == module.groups)

        # not recurse
        for n, p in module.named_parameters(recurse=False):
            param_group = {"params": [p]}
            if not p.requires_grad:
                params.append(param_group)
                continue
            if self.bypass_duplicate and self._is_in(param_group, params):
                warnings.warn(f'{prefix} is duplicate. It is skipped since '
                              f'bypass_duplicate={self.bypass_duplicate}')
                continue

            # there exists two independent cases
            # one case: if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in self.custom_keys:
                if key in f'{prefix}.{n}':
                    is_custom = True
                    lr_mult = self.custom_keys[key].get("lr_mult", 1.)
                    param_group["lr"] = self.base_lr * lr_mult
                    if self.base_weightdecay is not None:
                        wd_mult = self.custom_keys[key].get("decay_mult", 1.)
                        param_group['weight_decay'] = self.base_weightdecay * wd_mult
                    break

            # another case: if no custom keys, follow `bias_lr_mult`, etc.
            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if n == "bias" and not (is_norm or is_dcn_module):
                    param_group["lr"] = self.base_lr * self.bias_lr_mult

                if (prefix.find('conv_offset') != -1 and is_dcn_module and isinstance(module, torch.nn.Conv2d)):
                    # deal with both dcn_offset's bias & weight
                    param_group["lr"] = self.base_lr * self.dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm:
                        param_group['weight_decay'] = self.base_wd * self.norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group['weight_decay'] = self.base_wd * self.dwconv_decay_mult
                    # bias lr and decay
                    elif n == 'bias' and not is_dcn_module:
                        # TODO: current bias_decay_mult will have affect on DCN
                        param_group['weight_decay'] = self.base_wd * self.bias_decay_mult


            # No matter what case, add param_group
            params.append(param_group)
            # TODO: consider dcn flags, in this version we don't consider
            #  the effect of bias_decay_mult on DCN
            for children_name, children_module in module.named_children():
                child_prefix = f"{prefix}.{children_name}" if prefix else children_name
                self.adjust_lr_dw(
                    params,
                    children_module,
                    child_prefix,
                    is_dcn_module
                )