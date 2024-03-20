# Copyright (c) MDLDrugLib. All rights reserved.
import copy, math, warnings, torch
from typing import Union, List, Optional
import numpy as np
import torch.nn as nn
from torch import Tensor

from druglib.utils import Registry, build_from_cfg

INITIALIZERS = Registry("initializer")

def update_init_info(
        module:nn.Module,
        init_info:str,
) -> None:
    """
    Update the `_params_init_info` in the module if the value of parameters
        are changed.
    Args:
        module:nn.Module: The module of Pytorch with a user-defined attribute
            `_params_init_info` which records the initialization information.
        init_info:str: The string that describes the initialization.
    """
    assert hasattr(module, '_params_init_info'), \
    f'Can not find `_params_init_info` in {module}'

    for name, param in module.named_parameters():
        assert param in module._params_init_info, (
            f'Find a new :obj:`Parameter` '
            f'named `{name}` during executing the '
            f'`init_weights` of '
            f'`{module.__class__.__name__}`. '
            f'Please do not add or '
            f'replace parameters during executing the `init_weights`.'
        )
        # The parameter has been changed during executing the
        # `init_weights` of module.
        mean_value = param.data.mean()
        if module._params_init_info[param]['tmp_mean_value'] != mean_value:
            module._params_init_info[param]['init_info'] = init_info
            module._params_init_info[param]['tmp_mean_value'] = mean_value

def _initialize(
        module:nn.Module,
        cfg:dict,
        wholemodule:bool = False,
) -> None:
    """
    `wholemodule` flag is for override mode, there is no layer
        key in override and initializer will give init values
        for the whole module with the name in override.
    """
    func = build_from_cfg(
        cfg,
        INITIALIZERS,
    )
    func.wholemodule = wholemodule
    func(module)

def __initialize_override(
        module:nn.Module,
        override:Union[list, dict],
        cfg:dict,
) -> None:
    if not isinstance(override, (dict, list)):
        raise TypeError(
            f'`override` must be a dict or a list of dict, but got {type(override)}'
        )
    override = [override] if isinstance(override, dict) else override

    for override_ in override:
        cp_override = copy.deepcopy(override_)
        name = cp_override.pop('name', None)
        if name is None:
            raise ValueError(
                f'`override` must be contain the key "name", but got {cp_override}'
            )
        # if override only has name key, it means use args in init_cfg
        if not cp_override:
            cp_override.update(cfg)
        # if override has name key and other args except type key, it will raise error
        # Use `type` to pull predefined :obj:`INITIALIZERS`.
        elif 'type' not in cp_override.keys():
            raise ValueError(
                f'`override` need "type" key, but got {cp_override}'
            )
        if hasattr(module, name):
            _initialize(getattr(module, name), cp_override, wholemodule = True)
        else:
            raise RuntimeError(
                f'module did not have attribute {name}, '
                f'but init_cfg is {cp_override}.'
            )

def initialize(
        module:nn.Module,
        init_cfg:Union[dict, List[dict]],
):
    r"""
    Initialize a module.
    Args:
        module:nn.Module: the module will be initialized.
        init_cfg:Union[dict, List[dict]]: initialization config dict
            to define initializer. Druglib has implemented 6 initializers
            including ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
            ``Kaiming`` and ``Pretrained``.
    E.g.:
        >>> module = nn.Linear(2, 3, bias = True)
        >>> init_cfg = dict(type = 'Constant', layer = 'Linear', val = 1, bias = 2)
        >>> initialize(module, init_cfg = init_cfg)

        >>> module = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1, 2))
        >>> # define key ``layer`` for initializing layer with different
        >>> # configuration
        >>> init_cfg = [dict(type='Constant', layer = 'Conv1d', val = 1),
                        dict(type = 'Constant', layer = 'linear', val = 2)]
        >>> initialize(module, init_cfg)

        >>> # define key ``override`` to initialize some specific part in
        >>> # module. overrided obj must be module attributes.
        >>> class FooNet(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.feat = nn.Conv2d(3, 16, 3)
        >>>         self.reg = nn.Conv2d(16, 10, 3)
        >>>         self.cls = nn.Conv2d(16, 5, 3)
        >>> model = FooNet()
        >>> init_cfg =  [dict(type='Constant', layer = 'Conv2d', val = 1, bias = 2,
                        override = dict(type = 'Constant', name = 'reg', val = 3, bias = 4)]
        >>> initialize(module, init_cfg)

        >>> model = ResNet(depth = 50)
        >>> # Initialize weights wth the pretrained model from torchvision.
        >>> init_cfg = dict(type = 'Pretrained', checkpoint = 'torchvision://resnet50')
        >>> initialize(model, init_cfg=init_cfg)

        >>> # Initialize weights of a sub-module with the specific part of
        >>> # a pretrained model by using "prefix".
        >>> url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco'\
        >>>         '/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        >>> init_cfg = dict(type = 'Pretrained', checkpoint = url, prefix = 'backbone.')
        >>> initialize(model, init_cfg=init_cfg)
    """
    if not isinstance(init_cfg, (dict, list)):
        raise TypeError(
            f'`init_cfg` must be a dict or a list of dict, but got {type(init_cfg)}'
        )
    if isinstance(init_cfg, dict):
        init_cfg = [init_cfg]

    for cfg in init_cfg:

        cp_cfp = copy.deepcopy(cfg)
        override = cp_cfp.pop('override', None)
        _initialize(module, cp_cfp)

        if override is not None:
            cp_cfp.pop('layer', None)
            __initialize_override(module, override, cp_cfp)
        else:
            # All attributes in module have same initialization
            pass

def constant_init(
        module:nn.Module,
        val:Union[int, float],
        bias:Union[int, float] = 0,
) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def xavier_init(
        module:nn.Module,
        gain:Union[int, float] = 1,
        bias:Union[int, float] = 0,
        distribution:str = 'normal',
) -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain = gain)
        else:
            nn.init.xavier_normal_(module.weight, gain = gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def glorot(
    tensor: Tensor,
):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    tensor.data.uniform_(-stdv, stdv)

def glorot_init(
        module:nn.Module,
        bias:Union[int, float] = 0,
) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        value = module.weight
        glorot(value)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def glorot_orthogonal_init(
        module:nn.Module,
        scale: float = 2.0,
        bias:Union[int, float] = 0,
) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        value = module.weight
        nn.init.orthogonal_(value.data)
        scale /= ((value.size(-2) + value.size(-1)) * value.var())
        value.data *= scale.sqrt()
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def _standardize(kernel: Tensor):
    """Make sure that Var(W) = 1 and E[W] = 0"""
    eps = 1e-9
    if len(kernel) == 3:
        dim = (0, 1) # last dimension is output dimension
    else:
        dim = 1
    var, mu = torch.var_mean(kernel, dim = dim,
                             unbiased = True, keepdim = True)
    return (kernel - mu) / (var + eps) ** 0.5

def _he_orthogonal_init(
        tensor: Tensor,
):
    """
    Generate a weight matrix with variance according to He intialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = nn.init.orthogonal_(tensor)
    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor

def he_orthogonal_init(
        module:nn.Module,
        bias:Union[int, float] = 0,
) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        _he_orthogonal_init(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(
        module:nn.Module,
        mean:Union[int, float] = 0,
        std:Union[int, float] = 1,
        bias:Union[int, float] = 0,
) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean = mean, std = std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def uniform_init(
        module:nn.Module,
        lower:Union[int, float] = 0,
        upper:Union[int, float] = 1,
        bias:Union[int, float] = 0,
):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.uniform_(module.weight, lower, upper)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def trunc_normal_(
        tensor:Tensor,
        mean:Union[int, float],
        std:Union[int, float],
        a:Union[int, float],
        b:Union[int, float],
) -> Tensor:
    r"""
    Fills the input Tensor with values drawn from a truncated
        normal distribution. The values are effectively drawn from
        the normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^{2})`
        with values outside :math:`[a, b]` redrawn until they are within the bounds.
        The method used for generating the random values works best when :math:
        `a \leq \text{mean} \leq b`.
    Args:
        tensor:torch.Tensor: n-dimensional torch.Tensor.
        mean:Union[int, float]: The mean of the normal distribution.
        std:Union[int, float]: The standard deviation of the normal distribution.
        a:Union[int, float]: The minimum cutoff value.
        b:Union[int, float]: The maximum cutoff value.
    """
    assert a < b
    # copy from mmdetection (https://github.com/open-mmlab/mmdetection)
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init_trunc_normal_.'
            'The distribution of values may be incorrect.',
            stacklevel = 2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values.
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((a - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate
        # to [2 * lower -1, 2 * upper - 1]
        tensor.uniform_(2 * lower -1, 2 * upper - 1)

        # Use inverse cdf function for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp(min = a, max = b)

        return tensor

def trunc_normal_init(
        module:nn.Module,
        mean:Union[int, float] = 0,
        std:Union[int, float] = 1,
        a:Union[int, float] = -2,
        b:Union[int, float] = 2,
        bias:Union[int, float] = 0,
) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(
            tensor = module.weight,
            mean = mean,
            std = std,
            a = a,
            b = b,
        )
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(
        module:nn.Module,
        a:Union[int, float] = 0,
        mode:str = 'fan_out',
        nonlinearity:str = 'relu',
        bias:Union[int, float] = 0,
        distribution:str = 'normal',
) -> None:
    assert distribution.lower() in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution.lower() == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight,
                a = a,
                mode = mode,
                nonlinearity = nonlinearity,
            )
        else:
            nn.init.kaiming_normal_(
                module.weight,
                a=a,
                mode=mode,
                nonlinearity=nonlinearity,
            )
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_uniform_init(
        *args, **kwargs
):
    if 'distribution' in kwargs.keys():
        kwargs['distribution'] = 'uniform'
    kaiming_init(*args, **kwargs, distribution = 'uniform')

def caffe2_xavier_init(
        module:nn.Module,
        bias:Union[int, float] = 0,
) -> None:
    # Acknowledgment to FAIR's internel code
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    kaiming_init(
        module,
        a = 1,
        mode = 'fan_in',
        nonlinearity = 'leaky_relu',
        bias = bias,
        distribution = 'uniform',
    )

def bias_init_with_prob(
        prior_prob:float,
) -> float:
    """
    Initialize conv/fc bias value according to a given probability value.
    """
    return float(-np.log((1 - prior_prob) / prior_prob))

def _get_bases_name(
        module:nn.Module,
) -> List[str]:
    return [b.__name__ for b in module.__class__.__bases__]

class BaseInit(object):
    """
    Use * to define positional args: bias, bias_prob, layer.
    Initialize module parameters.
    Args:
        bias:Union[int, float]: The value to fill the bias. Defaults to 0.
        bias_prob:Optional[float]:The probability for bias initialization. Defaults to None.
        layer:Union[List[str], str, None]: The layer will be initialized. Defaults to None.
    """
    def __init__(
            self,
            *,
            bias:Union[int, float] = 0,
            bias_prob:Optional[float] = None,
            layer:Union[List[str], str, None] = None
    ):
        self.wholemodule = False
        # check type phase
        if not isinstance(bias, (int, float)):
            raise TypeError(
                f'bias must be a number, but got a {type(bias)}.'
            )

        if bias_prob is not None:
            if not isinstance(bias_prob, float):
                raise TypeError(
                    f'`bias_prob` must be float, but got {type(bias_prob)}.'
                )

        if layer is not None:
            if not isinstance(layer, (str, list)):
                raise TypeError(
                    f'layer must be a str or a list of str, but got a {type(layer)}.'
                )
        else:
            layer = []

        if bias_prob is not None:
            self.bias = bias_init_with_prob(bias_prob)
        else:
            self.bias = bias
        self.layer = [layer] if isinstance(layer, str) else layer

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}, bias = {self.bias}'
        return info

@INITIALIZERS.register_module(name = 'Constant')
class ConstantInit(BaseInit):
    """
    Initialize module parameters with constant values.
    Args:
        val:Union[int, float]: The value to fill the weights in module with.
        other args: as the BaseInit described.
    """

    def __init__(
            self,
            val:Union[int, float],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.val = val

    def __call__(
            self,
            module:nn.Module,
    ) -> None:
        def init(
                m:nn.Module
        ) -> None:
            if self.wholemodule:
                constant_init(m, self.val, self.bias)
            else:
                layername = m.__class__.__name__
                basename = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basename)):
                    constant_init(m, self.val, self.bias)
        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(
                module, init_info = self._get_init_info()
            )

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}: val = {self.val}, bias = {self.bias}'
        return info

@INITIALIZERS.register_module(name = 'Xavier')
class XavierInit(BaseInit):

    """
    Args:
        gain:Union[int, float]: An optional scaling factor. Defaults to 1.
        distribution:str: Distribution either be `normal` or `uniform`. Defaults to `normal`.
    """
    def __init__(
            self,
            gain:Union[int, float] = 1,
            distribution:str = 'normal',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.gain = gain
        self.distribution = distribution

    def __call__(
            self,
            module:nn.Module,
    ):
        def init(
                m: nn.Module
        ) -> None:
            if self.wholemodule:
                xavier_init(m, self.gain, self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                basename = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basename)):
                    xavier_init(m, self.gain, self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(
                module, init_info = self._get_init_info()
            )

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}: gain = {self.gain}, ' \
               f'distribution = {self.distribution}, bias = {self.bias}'
        return info

@INITIALIZERS.register_module(name = 'Normal')
class NormalInit(BaseInit):
    """
    Initialize module parameters with the values drawn from the normal distribution.
    Args:
        mean:Union[int, float]: The mean of the normal distribution. Defaults to 0.
        std:Union[int, float]: The standard deviation of the normal distribution. Defaults to 1.
    """
    def __init__(
            self,
            mean:Union[int, float] = 0,
            std:Union[int, float] = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def __call__(
            self,
            module:nn.Module,
    ) -> None:
        def init(
                m: nn.Module
        ) -> None:
            if self.wholemodule:
                normal_init(m, self.mean, self.std, self.bias)
            else:
                layername = m.__class__.__name__
                basename = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basename)):
                    normal_init(m,  self.mean, self.std, self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(
                module, init_info = self._get_init_info()
            )

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}: mean = {self.mean}, ' \
               f'standard deviation = {self.std}, bias = {self.bias}'
        return info

@INITIALIZERS.register_module(name = 'Uniform')
class UniformInit(BaseInit):
    """
    Initialize module parameters with values drawn from the uniform distribution.
    Args:
        a:Union[int, float]: The lower bound of the uniform distribution. Defaults to 0.
        b:Union[int, float]: The upper bound of the uniform distribution. Defaults to 1.
    """

    def __init__(
            self,
            a:Union[int, float] = 0,
            b:Union[int, float] = 1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def __call__(
            self,
            module:nn.Module,
    ) -> None:
        def init(
                m: nn.Module
        ) -> None:
            if self.wholemodule:
                uniform_init(m, self.a, self.b, self.bias)
            else:
                layername = m.__class__.__name__
                basename = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basename)):
                    uniform_init(m, self.a, self.b, self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(
                module, init_info = self._get_init_info()
            )

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}: a = {self.a}, ' \
               f'b = {self.b}, bias = {self.bias}'
        return info

@INITIALIZERS.register_module(name = 'TruncNormal')
class TruncNormalInit(BaseInit):
    """Initialize module parameters with the values drawn from the normal distribution
        with outside [a, b].
    Args:
        mean:Union[int, float]: The mean of the normal distribution. Defaults to 0.
        std:Union[int, float]: The standard deviation of the normal distribution. Defaults to 1.
        a:Union[int, float]: The minimum cutoff value. Defaults to -2.
        b:Union[int, float]: The maximum cutoff value. Defaults to 2.
    """

    def __init__(
            self,
            mean:Union[int, float] = 0,
            std: Union[int, float] = 1,
            a: Union[int, float] = -2,
            b: Union[int, float] = 2,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

    def __call__(
            self,
            module:nn.Module,
    ) -> None:
        def init(
                m: nn.Module
        ) -> None:
            if self.wholemodule:
                trunc_normal_init(m, self.mean, self.std, self.a, self.b, self.bias)
            else:
                layername = m.__class__.__name__
                basename = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basename)):
                    trunc_normal_init(m, self.mean, self.std, self.a, self.b, self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(
                module, init_info = self._get_init_info()
            )

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}: mean = {self.mean}, ' \
               f'standard deviation = {self.std}, a = {self.a}, b = {self.b}, bias = {self.bias}'
        return info

@INITIALIZERS.register_module(name = 'Kaiming')
class KaimingInit(BaseInit):
    r"""Initialize module parameters with the values according to the method
        described in `Delving deep into rectifiers: Surpassing human-level
        performance on ImageNet classification - He, K. et al. (2015).
        <https://www.cv-foundation.org/openaccess/content_iccv_2015/
        papers/He_Delving_Deep_into_ICCV_2015_paper.pdf>`

    Args:
        a:Union[int, float]: The negative slope of the rectifier used after this
            layer (only used with ``'leaky_relu'``). Defaults to 0.
        mode:str: Either ``'fan_in'`` or ``'fan_out'``. Choosing
            ``'fan_in'`` preserves the magnitude of the variance of the weights
            in the forward pass. Choosing ``'fan_out'`` preserves the
            magnitudes in the backwards pass. Defaults to ``'fan_out'``.
        nonlinearity:str: The non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` .
            Defaults to 'relu'.
        distribution:str: Distribution either be ``'normal'`` or
            ``'uniform'``. Defaults to ``'normal'``.
    """


    def __init__(
            self,
            a:Union[int, float] = 0,
            mode:str = 'fan_out',
            nonlinearity:str = 'relu',
            distribution:str = 'normal',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution

    def __call__(
            self,
            module:nn.Module,
    ) -> None:
        def init(m):
            if self.wholemodule:
                kaiming_init(m, self.a, self.mode, self.nonlinearity,
                             self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    kaiming_init(m, self.a, self.mode, self.nonlinearity,
                                 self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info = self._get_init_info())

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}: a = {self.a}, mode = {self.mode}, ' \
               f'nonlinearity = {self.nonlinearity}, ' \
               f'distribution = {self.distribution}, bias = {self.bias}'
        return info

@INITIALIZERS.register_module(name = 'Caffe2Xavier')
class Caffe2XavierInit(KaimingInit):
    def __init__(self, **kwargs):
        super().__init__(
            a = 1,
            mode = 'fan_in',
            nonlinearity = 'leaky_relu',
            distribution = 'uniform',
            **kwargs
        )

    def __call__(
            self,
            module:nn.Module,
    ) -> None:
        super().__call__(module)

@INITIALIZERS.register_module(name = 'Pretrained')
class PretrainedInit(object):
    """
    Initialize module by loading a pretrained model.

    Args:
        checkpoint:str: The checkpoint file of the pretrained model should
            be load.
        prefix:Optional[str]: The prefix of a sub-module in the pretrained
            model. it is for loading a part of the pretrained model to
            initialize. For example, if we would like to only load the
            backbone of a detector model, we can set ``prefix='backbone.'``.
            Defaults to None.
        map_location:str: Map tensors into proper locations.
    """
    def __init__(
            self,
            checkpoint:str,
            prefix:Optional[str] = None,
            map_location:Optional[str] = None
    ):
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.map_location = map_location

    def __call__(
            self,
            module:nn.Module,
    ) -> None:

        raise NotImplementedError

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info = self._get_init_info())

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}: load from {self.checkpoint}'
        return info
