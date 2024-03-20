# Copyright (c) MDLDrugLib. All rights reserved.
"""
Most of the code adopted from
https://github.com/gcorso/DiffDock/blob/main/utils/so3.py
"""
import os
from typing import Union
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import io

MIN_EPS, MAX_EPS, N_EPS = 0.01, 2, 1000
X_N = 2000

"""
Preprocessing for the SO(3) sampling and score computations, truncated infinite series are computed and then
cached to memory, therefore the precomputation is only run the first time the repository is run on a machine.
Shameless steal from: https://github.com/gcorso/DiffDock/blob/main/utils/so3.py
"""

omegas = np.linspace(0, np.pi, X_N + 1)[1:]

def _compose(r1, r2):
    return Rotation.from_matrix(Rotation.from_rotvec(r1).as_matrix() @ Rotation.from_rotvec(r2).as_matrix()).as_rotvec()


def _expansion(omega, eps, L = 2000):
    """
    The summation term only
    f(\omega) = \sum_{l=0}^{\infty}(2l + 1)exp(-l(l+1)\sigma^2)\frac{sin((l + 0.5))\omega}{sin(\omega/2)}
    Here is truncated infinite series.
    """
    p = 0
    for l in range(L):
        p += (2 * l + 1) * np.exp(-l * (l + 1) * eps ** 2) * np.sin(omega * (l + 1 / 2)) / np.sin(omega / 2)
    return p


def _density(expansion, omega, marginal = True):
    """
    expansion: precalculated from :func:`_expansion`
    marginal: If True, density over [0, pi]; else over SO(3)
    """
    if marginal:
        return expansion * (1 - np.cos(omega)) / np.pi
    else:
        return expansion / 8 / np.pi ** 2  # the constant factor doesn't affect any actual calculations though


def _score(exp, omega, eps, L = 2000):
    """
    score of density over SO(3)
    """
    dSigma = 0
    for l in range(L):
        hi = np.sin(omega * (l + 1 / 2))
        dhi = (l + 1 / 2) * np.cos(omega * (l + 1 / 2))
        lo = np.sin(omega / 2)
        dlo = 1 / 2 * np.cos(omega / 2)
        dSigma += (2 * l + 1) * np.exp(-l * (l + 1) * eps ** 2) * (lo * dhi - hi * dlo) / lo ** 2
    return dSigma / exp


resource_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../resources'))
diffusion_dir = Path(resource_dir) / 'diffusion'
os.makedirs(diffusion_dir, exist_ok = True)

lmdb_mode = True
if os.path.exists(diffusion_dir / 'so3_series.lmdb'):
    env = io._load_lmdb(diffusion_dir / 'so3_series.lmdb')
    # _omegas_array = io._load_lmdb_data(env, 'so3_omegas_array2')
    # _cdf_vals =  io._load_lmdb_data(env, 'so3_cdf_vals2')
    # _score_norms =  io._load_lmdb_data(env, 'so3_score_norms2')
    # _exp_score_norms =  io._load_lmdb_data(env, 'so3_exp_score_norms2')

    _omegas_array, _cdf_vals, _score_norms, _exp_score_norms = io._load_lmdb_data(
        env, [
            'so3_omegas_array2',
            'so3_cdf_vals2',
            'so3_score_norms2',
            'so3_exp_score_norms2',
        ]
    )
elif os.path.exists(diffusion_dir / 'so3_omegas_array2.npy'):
    _omegas_array =  io._load(diffusion_dir / 'so3_omegas_array2.npy')
    _cdf_vals =  io._load(diffusion_dir / 'so3_cdf_vals2.npy')
    _score_norms =  io._load(diffusion_dir / 'so3_score_norms2.npy')
    _exp_score_norms =  io._load(diffusion_dir / 'so3_exp_score_norms2.npy')
else:
    _eps_array = 10 ** np.linspace(np.log10(MIN_EPS), np.log10(MAX_EPS), N_EPS)
    _omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]

    _exp_vals = np.asarray([_expansion(_omegas_array, eps) for eps in _eps_array])
    _pdf_vals = np.asarray([_density(_exp, _omegas_array, marginal = True) for _exp in _exp_vals])
    _cdf_vals = np.asarray([_pdf.cumsum() / X_N * np.pi for _pdf in _pdf_vals])
    _score_norms = np.asarray([_score(_exp_vals[i], _omegas_array, _eps_array[i]) for i in range(len(_eps_array))])
    # \nabla_{x_t}log p(x_t | x_0)
    _exp_score_norms = np.sqrt(np.sum(_score_norms**2 * _pdf_vals, axis = 1) / np.sum(_pdf_vals, axis = 1) / np.pi)

    if lmdb_mode:
        io._save_lmdb(
            diffusion_dir / 'so3_series.lmdb',
            {
                'so3_omegas_array2': _omegas_array,
                'so3_cdf_vals2': _cdf_vals,
                'so3_score_norms2': _score_norms,
                'so3_exp_score_norms2': _exp_score_norms,
            }
        )
    else:
        io._save(diffusion_dir / 'so3_omegas_array2.npy', _omegas_array)
        io._save(diffusion_dir / 'so3_cdf_vals2.npy', _cdf_vals)
        io._save(diffusion_dir / 'so3_score_norms2.npy', _score_norms)
        io._save(diffusion_dir / 'so3_exp_score_norms2.npy', _exp_score_norms)


def sample(eps):
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min = 0, a_max = N_EPS - 1)

    # x = np.random.rand()
    x = np.random.default_rng().uniform(0, 1)
    return np.interp(x, _cdf_vals[eps_idx], _omegas_array)


def sample_vec(eps):
    # x = np.random.randn(3)
    x = np.random.default_rng().normal(0, 1, 3)
    x /= np.linalg.norm(x)
    return x * sample(eps)


def score_vec(eps, vec):
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min = 0, a_max = N_EPS - 1)

    om = np.linalg.norm(vec)
    return np.interp(om, _omegas_array, _score_norms[eps_idx]) * vec / om


def score_norm(eps: Union[torch.Tensor, np.ndarray]):
    if torch.is_tensor(eps):
        eps = eps.cpu().numpy()
    eps_idx = (np.log10(eps) - np.log10(MIN_EPS)) / (np.log10(MAX_EPS) - np.log10(MIN_EPS)) * N_EPS
    eps_idx = np.clip(np.around(eps_idx).astype(int), a_min = 0, a_max = N_EPS - 1)
    return torch.from_numpy(_exp_score_norms[eps_idx]).float()