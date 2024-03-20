# Copyright (c) MDLDrugLib. All rights reserved.
"""
Most of the code adopted from
https://github.com/gcorso/DiffDock/blob/main/utils/torus.py
"""
import os
import tqdm
from pathlib import Path
from typing import Union
import numpy as np
import torch

from . import io

"""
Preprocessing for the SO(2)/torus sampling and score computations, truncated infinite series are computed and then
cached to memory, therefore the precomputation is only run the first time the repository is run on a machine.
Shameless steal from: https://github.com/gcorso/DiffDock/blob/main/utils/torus.py
"""

def p(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


def grad(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi

resource_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources'))
diffusion_dir = Path(resource_dir) / 'diffusion'
os.makedirs(diffusion_dir, exist_ok = True)

lmdb_mode = True
if os.path.exists(diffusion_dir / 'torus_series.lmdb'):
    env = io._load_lmdb(diffusion_dir / 'torus_series.lmdb')
    # p_ = io._load_lmdb_data(env, 'torus_p')
    # score_ = io._load_lmdb_data(env, 'torus_score')

    p_, score_ = io._load_lmdb_data(
        env, ['torus_p', 'torus_score']
    )
elif os.path.exists(diffusion_dir / 'torus_p.npy'):
    p_ = io._load(diffusion_dir / 'torus_p.npy')
    score_ = io._load(diffusion_dir / 'torus_score.npy')
else:
    p_ = p(x, sigma[:, None], N=100)
    score_ = grad(x, sigma[:, None], N=100) / p_
    if lmdb_mode:
        io._save_lmdb(
            diffusion_dir / 'torus_series.lmdb',
            {
                'torus_p': p_,
                'torus_score': score_,
            }
        )
    else:
        io._save(diffusion_dir / 'torus_p.npy', p_)
        io._save(diffusion_dir / 'torus_score.npy', score_)


def score(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]


def p(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out


score_norm_ = score(
    sample(sigma[None].repeat(10000, 0).flatten()),
    sigma[None].repeat(10000, 0).flatten()
).reshape(10000, -1)
score_norm_ = (score_norm_ ** 2).mean(0)


def score_norm(sigma: Union[torch.Tensor, np.ndarray]):
    if torch.is_tensor(sigma):
        sigma = sigma.cpu().numpy()
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]