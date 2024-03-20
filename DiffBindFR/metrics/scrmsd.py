# Copyright (c) MDLDrugLib. All rights reserved.
import torch
from torch import Tensor

from druglib.utils.obj import protein_constants as pc


def make_altern_atom14(
        atom14_pos: Tensor,
        atom14_mask: Tensor,
        sequence: Tensor,
):
    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [pc.restype_1to3[res] for res in pc.restypes]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {
        res: torch.eye(
            14,
            dtype = atom14_pos.dtype,
            device = atom14_pos.device,
        )
        for res in restype_3
    }
    for resname, swap in pc.residue_atom_renaming_swaps.items():
        correspondences = torch.arange(
            14, device = atom14_pos.device
        )
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = pc.restype_name_to_atom14_names[resname].index(
                source_atom_swap
            )
            target_index = pc.restype_name_to_atom14_names[resname].index(
                target_atom_swap
            )
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = atom14_pos.new_zeros((14, 14))
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix

    renaming_matrices = torch.stack(
        [all_matrices[restype] for restype in restype_3]
    )

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[sequence]
    renaming_transform = renaming_transform.view(*((1, ) * (atom14_pos.dim() - 3) + renaming_transform.shape))

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_atom14_pos = torch.einsum(
        "...rac,...rab->...rbc", atom14_pos, renaming_transform
    )
    alternative_atom14_mask = torch.einsum(
        "...ra,...rab->...rb", atom14_mask.float(), renaming_transform
    )
    return alternative_atom14_pos, alternative_atom14_mask


def sidechain_rmsd(
        pred_atom14: Tensor, # (..., N, 14, 3)
        target_atom14: Tensor, # (N, 14, 3)
        target_atom14_mask: Tensor, # (N, 14)
        sequence: Tensor, # (N,)
        eps: float = 1e-6,
):
    target_atom14 = target_atom14.view(*((1, ) * (pred_atom14.dim() - 3) + target_atom14.shape))
    target_atom14_mask = target_atom14_mask.view(*((1,) * (pred_atom14.dim() - 3) + target_atom14_mask.shape))
    sc_atm_mask = target_atom14_mask[..., 5:]
    sc_atm_pred = pred_atom14[..., 5:, :] * sc_atm_mask[..., None]
    sc_atm_target = target_atom14[..., 5:, :] * sc_atm_mask[..., None]
    alternative_atom14_target, alternative_atm_mask = make_altern_atom14(
        target_atom14, target_atom14_mask, sequence,
    )
    altern_atm_mask = alternative_atm_mask[..., 5:]
    altern_atm_target = alternative_atom14_target[..., 5:, :] * altern_atm_mask[..., None]

    dist_square = ((sc_atm_target - sc_atm_pred) ** 2).sum(dim = (-2, -1))
    altern_dist_square = ((altern_atm_target - sc_atm_pred) ** 2).sum(dim = (-2, -1))
    dist_square = torch.minimum(dist_square, altern_dist_square)
    sc_res_mask = sc_atm_mask.any(dim = -1)
    deno = sc_atm_mask.sum(dim = -1)
    rmsd = torch.sqrt(dist_square / (deno + eps)) * sc_res_mask
    rmsd = rmsd.sum(dim = -1) / sc_res_mask.sum(dim = -1)

    return rmsd