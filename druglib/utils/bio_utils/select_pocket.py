# Copyright (c) MDLDrugLib. All rights reserved.
"""Differentiable binding site selection function using PyTorch"""
from typing import (
    Optional, Tuple,
)
import torch
from torch import FloatTensor, BoolTensor

from ..torch_utils import batched_gather_assign


def _select_min(
        pocket_res_mask: BoolTensor,
        dists: FloatTensor,
) -> BoolTensor:
    """
    Safe function to keep at least minimal distance residue will be selected
    Args:
        pocket_res_mask: shape (*, N)
        dists: shape (*, N)
    """
    min_id = torch.argmin(dists, dim = -1)
    pocket_res_mask = batched_gather_assign(
        pocket_res_mask, min_id,
        True, -1, len(pocket_res_mask.shape[:-1])
    )
    return pocket_res_mask

def _max_neig_trunc(
        pocket_res_mask: BoolTensor,
        dists: FloatTensor,
        max_neighbors: int = 32,
        big_value: float = 1e20,
) -> BoolTensor:
    """
    Truncate residue mask to maximal neighborhoods
    Args:
        pocket_res_mask: shape (*, N)
        dists: shape (*, N)
    """
    reverse = torch.logical_not(pocket_res_mask)
    dists[reverse] = big_value
    sort = torch.sort(dists, dim = -1)[1]
    sort = sort[..., :max_neighbors]
    max_neighbors_mask = batched_gather_assign(
        torch.zeros_like(pocket_res_mask),
        sort, True, -1, len(pocket_res_mask.shape[:-1])
    )
    pocket_res_mask = torch.logical_and(
        pocket_res_mask,
        max_neighbors_mask,
    )
    return pocket_res_mask

def select_bs(
        lig_pos: FloatTensor,
        all_atom_positions: FloatTensor,
        all_atom_mask: FloatTensor,
        lig_mask: Optional[FloatTensor] = None,
        cutoff: float = 10.0,
        max_neighbors: Optional[int] = None,
        big_value: float = 1e20,
):
    """
    Select pocket around ligand atoms by cutoff.
    Args:
        lig_pos: shape (*, N_l, 3). Ligand positions
        all_atom_positions: shape (*, N_res, M, 3)
        all_atom_mask: shape (*, N_res, M). Existing atom mask
        lig_mask: shape (*, N_l) or None. Ligand atoms mask. If None, set ones.
        cutoff: Binding site selection radius (any ligand heavy atoms to residue atoms).
    Returns:
        BoolTensor with shape (*, N) indicating which are pocket residues
    """
    if lig_mask is None:
        lig_mask = lig_pos.new_ones(lig_pos.shape[:-1])
    lig_mask = lig_mask.bool()
    all_atom_mask = all_atom_mask.bool()

    # (*, N_res, M, 3) * (*, N_l, 3) -> (*, N_res, M, N_l)
    dist = torch.sum(
        (all_atom_positions[..., None, :] - lig_pos[..., None, None, :, :]) ** 2,
        dim = -1,
    )
    mask = all_atom_mask[..., None] * lig_mask[..., None, None, :]
    dist = torch.maximum(dist, (torch.logical_not(mask)).float() * big_value)
    # (*, N_res)
    per_res_mindist = torch.amin(dist, dim = (-2, -1))
    # reduce to (*, N_res,) by using any one distance per residue <= bs_radius
    res_mask = (per_res_mindist <= cutoff ** 2)  # the nearest atom within the cutoff
    res_mask = _select_min(res_mask, per_res_mindist)
    if max_neighbors is not None:
        res_mask = _max_neig_trunc(
            res_mask,
            per_res_mindist,
            max_neighbors = max_neighbors,
        )

    return res_mask

def select_bs_any(
        lig_pos: FloatTensor,
        all_atom_positions: FloatTensor,
        all_atom_mask: FloatTensor,
        *args,
        lig_mask: Optional[FloatTensor] = None,
        cutoff: float = 10.0,
        max_neighbors: Optional[int] = None,
        **kwargs
) -> BoolTensor:
    assert all_atom_positions.ndim > 2 and all_atom_positions.size(-1) == 3, \
        f'all_atom_positions shape must be (*, N, M, 3), but got {all_atom_positions.shape}'
    assert all_atom_mask.ndim > 1, \
        f'all_atom_mask shape must be (*, N, M), but got {all_atom_mask.shape}'
    assert lig_pos.ndim > 1 and lig_pos.size(-1) == 3, \
        f'all_atom_positions shape must be (*, N, 3), but got {lig_pos.shape}'
    res_mask = select_bs(
        lig_pos = lig_pos,
        all_atom_positions = all_atom_positions,
        all_atom_mask = all_atom_mask,
        lig_mask = lig_mask,
        cutoff = cutoff,
        max_neighbors = max_neighbors,
    )
    return res_mask

def select_bs_atoms(
        lig_pos: FloatTensor,
        all_atom_positions: FloatTensor,
        all_atom_mask: FloatTensor,
        atoms_id: Tuple[int, ...],
        *args,
        lig_mask: Optional[FloatTensor] = None,
        cutoff: float = 10.0,
        max_neighbors: Optional[int] = None,
        **kwargs
) -> BoolTensor:
    assert all_atom_positions.ndim > 2 and all_atom_positions.size(-1) == 3, \
        f'all_atom_positions shape must be (*, N, M, 3), but got {all_atom_positions.shape}'
    assert all_atom_mask.ndim > 1, \
        f'all_atom_mask shape must be (*, N, M), but got {all_atom_mask.shape}'
    assert lig_pos.ndim > 1 and lig_pos.size(-1) == 3, \
        f'all_atom_positions shape must be (*, N, 3), but got {lig_pos.shape}'
    all_atom_positions = all_atom_positions[..., atoms_id, :]
    all_atom_mask = all_atom_mask[..., atoms_id]
    res_mask = select_bs(
        lig_pos = lig_pos,
        all_atom_positions = all_atom_positions,
        all_atom_mask = all_atom_mask,
        lig_mask = lig_mask,
        cutoff = cutoff,
        max_neighbors = max_neighbors,
    )
    return res_mask

def select_bs_centroid(
        lig_pos: FloatTensor,
        all_atom_positions: FloatTensor,
        all_atom_mask: FloatTensor,
        *args,
        lig_mask: Optional[FloatTensor] = None,
        cutoff: float = 10.0,
        max_neighbors: Optional[int] = None,
        **kwargs
):
    centroids = all_atom_positions
    all_atom_mask = all_atom_mask.bool().any(dim = -1)
    assert centroids.ndim > 1 and centroids.size(-1) == 3, \
        f'all_atom_positions shape must be (*, N, 3), but got {centroids.shape}'
    assert all_atom_mask.ndim > 0, \
        f'all_atom_mask shape must be (*, N), but got {all_atom_mask.shape}'
    assert lig_pos.ndim > 1 and lig_pos.size(-1) == 3, \
        f'all_atom_positions shape must be (*, N, 3), but got {lig_pos.shape}'
    centroids = centroids[..., None, :]
    res_mask = select_bs(
        lig_pos = lig_pos,
        all_atom_positions = centroids,
        all_atom_mask = all_atom_mask,
        lig_mask = lig_mask,
        cutoff = cutoff,
        max_neighbors = max_neighbors,
    )
    return res_mask


if __name__ == '__main__':
    import random
    batch = (32, )
    lig_pos = torch.randn(batch + (28, 3))
    all_atom_positions = torch.randn(batch + (128, 37, 3))
    all_atom_mask = torch.randn(all_atom_positions.shape[:-1]) > 0
    all_atom_mask = all_atom_mask.float()
    lig_mask = (torch.randn(lig_pos.shape[:-1]) > 0).float()
    res_mask = select_bs_any(
        lig_pos,
        all_atom_positions,
        all_atom_mask,
        lig_mask if random.random() > 0.5 else None,
        cutoff = 10,
    )
    print(res_mask.shape)
    print(res_mask)