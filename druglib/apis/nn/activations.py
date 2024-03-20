# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional
from torch.types import Number
import math
from packaging import version
from functools import partial

import torch
from torch import nn


def gelu_python(x):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

class gelu(nn.Module):
    def __init__(self, mode: str = 'plain'):
        super(gelu, self).__init__()
        assert isinstance(mode, str) and \
        mode.lower() in ['plain', 'quick', 'fast', 'new']
        mode = mode.lower()
        self.gelu_mode = mode
        if mode == 'plain':
            if version.parse(torch.__version__) < version.parse("1.4"):
                act = gelu_python
            else:
                act = nn.functional.gelu
        elif mode == 'quick':
            act = quick_gelu
        elif mode == 'fast':
            act = gelu_fast
        elif mode == 'new':
            act = gelu_new
        self.act = act

    def forward(self, x):
        return self.act(x)

def _silu_python(x):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """
    return x * torch.sigmoid(x)

class silu(nn.Module):
    def __init__(self, scale: float = 1.0):
        super(silu, self).__init__()
        if version.parse(torch.__version__) < version.parse("1.7"):
            act = _silu_python
        else:
            act = nn.functional.silu
        self.act = act
        self.scale = scale

    def forward(self, x):
        return self.act(x) / self.scale

class hswish(nn.Module):
    """
    Hard Swish Module.
    This module applies the hard swish function:
        math:
            hswish(x) = x * ReLU6(x + 3) / 6
    """
    def __init__(
            self,
            inplace: bool = False
    ):
        super(hswish, self).__init__()
        self.act = nn.ReLU6(inplace = inplace)

    def forward(self, x):
        return x * self.act(x + 3) / 6

def _mish_python(x):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """
    return x * torch.tanh(nn.functional.softplus(x))

class mish(nn.Module):
    def __init__(self, *args, **kwargs):
        super(mish, self).__init__()
        if version.parse(torch.__version__) < version.parse("1.9"):
            act = _mish_python
        else:
            act = nn.functional.mish
        self.act = act

    def forward(self, x):
        return self.act(x)

class clamp(nn.Module):
    def __init__(
            self,
            min: Optional[Number] = None,
            max: Optional[Number] = None,
    ):
        super(clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

ACT2FN = {
    "relu": nn.ReLU,
    "silu": silu,
    "ssilu": partial(silu, scale = 0.6),
    "swish": silu,
    "gelu": gelu,
    "tanh": nn.Tanh,
    "gelu_new": partial(gelu, mode = 'new'),
    "gelu_fast": partial(gelu, mode = 'fast'),
    "quick_gelu": partial(gelu, mode = 'quick'),
    "mish": mish,
    "linear": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "leakyrelu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "rrelu": nn.RReLU,
    "relu6": nn.ReLU6,
    "elu": nn.ELU,
    "clip": clamp,
    "hswish": hswish,
    "glu": nn.GLU,
    "Softplus": nn.Softplus,
    "selu": nn.SELU,
    "celu": nn.CELU,
}


def get_activation(activation_string: str):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")