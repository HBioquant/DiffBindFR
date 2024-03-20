# Copyright (c) MDLDrugLib. All rights reserved.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(
        timesteps: torch.FloatTensor,
        embed_dim: int = 64,
        max_positions: int = 10000,
) -> torch.FloatTensor:
    dtype = timesteps.dtype
    device = timesteps.device
    assert len(timesteps.shape) == 1
    half_dim = embed_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype = dtype, device = device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim = 1)
    if embed_dim % 2 == 1: # zero pad
        emb = F.pad(emb, (0, 1, 0, 0), mode = 'constant')
    assert emb.shape == (timesteps.shape[0], embed_dim)

    return emb

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """
    def __init__(
            self,
            embedding_size: int = 256,
            scale: float = 1.0,
    ):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad = False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim = -1)
        return emb

def get_timestep_embfunc(
        emb_type: str = 'sinusoidal',
        emb_dim: int = 64,
        emb_scale: int = 1000,
):
    if emb_type == 'sinusoidal':
        emb_func = (lambda x : sinusoidal_embedding(emb_scale * x, emb_dim))
    elif emb_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size = emb_dim, scale = emb_scale)
    else:
        raise NotImplementedError('Only support ddpm sinusoidal embedding or score matching Gaussian Fourier Projection.')

    return emb_func