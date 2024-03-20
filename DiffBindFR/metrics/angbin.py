# Copyright (c) MDLDrugLib. All rights reserved.
import torch
from torch import Tensor

from openfold.data.data_transforms import atom37_to_torsion_angles
from druglib.utils.torch_utils import batched_gather
from druglib.utils.obj import protein_constants as pc
from druglib.utils.obj import prot_math as pm


def angular_difference(
        target_angles: Tensor,
        predicted_angles: Tensor,
):
    """
    Args:
        target_angles: Tensor. sin \theta, cos \theta, shape (..., N, 2)
        predicted_angles: Tensor. sin \theta, cos \theta, shape (..., N, 2)

    Returns:
        Tensor: shape (..., N,)
    """
    target_radians = torch.atan2(target_angles[..., 0], target_angles[..., 1])
    predicted_radians = torch.atan2(predicted_angles[..., 0], predicted_angles[..., 1])
    diff_radians = torch.fmod(predicted_radians - target_radians + torch.pi, 2. * torch.pi) - torch.pi
    abs_diff_radians = torch.abs(diff_radians)
    clipped_diff_radians = torch.clamp(abs_diff_radians, min = 0, max = torch.pi)

    return clipped_diff_radians

def expand_font_fn(
        tensor: Tensor,
        exp_n: int,
) -> Tensor:
    return tensor.view(*((1, ) * exp_n + tensor.shape))

def expand_font_dim(
        tensor: Tensor,
        ref_tensor: Tensor,
        last_n: int = 3,
) -> Tensor:
    ndim = tensor.dim()
    exp_n = ref_tensor.dim() - last_n
    tensor = expand_font_fn(tensor, exp_n)
    tensor = tensor.repeat(ref_tensor.shape[:-last_n] + (1,) * ndim)
    return tensor

def chi_differ(
        pred_atom14: Tensor,  # (..., N, 14, 3)
        target_atom14: Tensor,  # (N, 14, 3)
        target_atom14_mask: Tensor,  # (N, 14)
        sequence: Tensor,  # (N,)
):
    mapper = pc.atoms14_to_atoms37_mapper[sequence]
    mapper = torch.LongTensor(mapper).to(pred_atom14.device)
    atom37_exists = pc.restype_atom37_mask[sequence]
    atom37_exists = target_atom14_mask.new_tensor(atom37_exists)
    target_atom37_mask = batched_gather(
        target_atom14_mask,
        mapper,
        dim = -1,
        batch_ndims = len(target_atom14_mask.shape[:-1])
    )
    target_atom37_mask = target_atom37_mask * atom37_exists
    target_atom37 = pm.atom14_to_atom37(target_atom14, mapper, target_atom37_mask)
    target_data = atom37_to_torsion_angles()(
        {
            'aatype': sequence,
            'all_atom_positions': target_atom37,
            'all_atom_mask': target_atom37_mask,
        }
    )
    target_tor_sin_cos = target_data['torsion_angles_sin_cos'][..., -4:, :]
    target_alt_tor_sin_cos = target_data['alt_torsion_angles_sin_cos'][..., -4:, :]
    torsion_angles_mask = target_data['torsion_angles_mask'][..., -4:]

    mapper = expand_font_dim(mapper, pred_atom14, 3)
    target_atom37_mask = expand_font_dim(target_atom37_mask, pred_atom14, 3)
    pred_atom37 = pm.atom14_to_atom37(pred_atom14, mapper, target_atom37_mask)
    sequence = expand_font_dim(sequence, pred_atom14, 3)
    pred_tor_sin_cos = atom37_to_torsion_angles()(
        {
            'aatype': sequence,
            'all_atom_positions': pred_atom37,
            'all_atom_mask': target_atom37_mask,
        }
    )['torsion_angles_sin_cos'][..., -4:, :]
    exp_n = pred_atom14.dim() - 3
    target_tor_sin_cos = expand_font_fn(target_tor_sin_cos, exp_n)
    target_alt_tor_sin_cos = expand_font_fn(target_alt_tor_sin_cos, exp_n)
    torsion_angles_mask = expand_font_fn(torsion_angles_mask, exp_n)

    pred_chi_differ = angular_difference(pred_tor_sin_cos, target_tor_sin_cos)
    aln_pred_chi_differ = angular_difference(pred_tor_sin_cos, target_alt_tor_sin_cos)
    pred_chi_differ = torch.minimum(pred_chi_differ, aln_pred_chi_differ)
    pred_chi_differ = pred_chi_differ * torsion_angles_mask

    return pred_chi_differ, torsion_angles_mask