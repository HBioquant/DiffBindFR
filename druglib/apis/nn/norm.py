# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch_scatter import scatter_mean
from druglib.utils import SyncBatchNorm
from druglib.data.typing import OptTensor


class CoordsNorm(nn.Module):
    def __init__(
            self,
            eps = 1e-6,
            scale_init = 1.,
    ):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coords):
        norm = coords.norm(dim = -1, keepdim = True)
        normed_coords = coords / norm.clamp_min(self.eps)
        return normed_coords * self.scale

class GraphNorm(torch.nn.Module):
    """
    Applies graph normalization over individual graphs as described in the
        `"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
        Training" <https://arxiv.org/abs/2009.03294>`_ paper
    """
    def __init__(
            self,
            in_dim: int,
            affine: bool = True,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(in_dim))
            self.bias = torch.nn.Parameter(torch.Tensor(in_dim))
            self.mean_scale = torch.nn.Parameter(torch.Tensor(in_dim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            self.register_buffer('mean_scale', torch.ones(in_dim, dtype = torch.float32))
        self.init_weights()

    def init_weights(self):
        if self.affine:
            self.weight.data.fill_(1.)
            self.bias.data.fill_(0.)
            self.mean_scale.data.fill_(1.)

    def forward(
            self,
            x: Tensor,
            batch: OptTensor = None,
    ) -> Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype = torch.long)

        batch_size = int(batch.max()) + 1

        mean = scatter_mean(x, batch, dim = 0, dim_size = batch_size)
        out = x - mean.index_select(0, batch) * self.mean_scale
        var = scatter_mean(out.pow(2), batch, dim = 0, dim_size = batch_size)
        std = (var + self.eps).sqrt().index_select(0, batch)

        if self.affine:
            return self.weight * out / std + self.bias
        return out / std

NORM2FN = {
    "BN": nn.BatchNorm2d,
    "BN2d": nn.BatchNorm2d,
    "BN1d": nn.BatchNorm1d,
    "BN3d": nn.BatchNorm3d,
    "SyncBN": SyncBatchNorm,
    "GN": nn.GroupNorm,
    "LN": nn.LayerNorm,
    "IN": nn.InstanceNorm2d,
    "IN2d": nn.InstanceNorm2d,
    "IN1d": nn.InstanceNorm1d,
    "IN3d": nn.InstanceNorm3d,
    "identity": nn.Identity,
    "Coords": CoordsNorm,
    "Graph": GraphNorm,


}


def get_norm(
        norm_string: str = 'identity',
        num_features: Optional[int] = None,
        requires_grad: bool = True,
        eps: float = 1e-9,
        **kwargs,
):
    if norm_string in NORM2FN:
        norm_layer = NORM2FN[norm_string]
        if norm_string == 'GN':
            assert 'num_groups' in kwargs
            layer = norm_layer(
                num_channels = num_features, eps = eps, **kwargs)
        elif norm_string == 'identity':
            layer = norm_layer()
        elif norm_string == 'Coords':
            layer = norm_layer(eps = eps, **kwargs)
        else:
            layer = norm_layer(num_features, eps = eps, **kwargs)
            if norm_string == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
                layer._specify_ddp_gpu_num(1)

        for param in layer.parameters():
            param.requires_grad = requires_grad

        return layer

    else:
        raise KeyError(f"function {norm_string} not found in NORM2FN mapping {list(NORM2FN.keys())}")