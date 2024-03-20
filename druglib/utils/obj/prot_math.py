# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, Tuple
from easydict import EasyDict as ed
import numpy as np

import torch
from torch import Tensor

from . import protein_constants as pc
from ..geometry_utils.utils import (
    make_rigid_transformation_4x4, parse_xrot_angle, radian2sincos,
    apply_inv_euclidean, rot_vec_around_x_axis, residue_frame,
)
from ..torch_utils import tree_map, batched_gather
from ..geometry_utils import aaframe


def to_pos14(
        aatype: np.ndarray,
        atom37_pos: np.ndarray,
        atom37_gt_exists: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Convert pos37 atom position repr to pos14 atom repr.
    Returns:
        pos14: np.ndarray. Shape [num_residue, 14, 3],
            possibly with missing atoms;
        mask14: np.ndarray. Shape [num_residue, 1];
    """
    n_res = aatype.shape[0]
    atoms37_to_atoms14 = pc.atoms37_to_atoms14_mapper[aatype]
    atom_positions = atom37_pos[
        np.arange(n_res).reshape(-1, 1),
        atoms37_to_atoms14]
    restype_atom14_mask = pc.restype_atom14_mask[aatype]
    if atom37_gt_exists is not None:
        restype_atom14_mask = atom37_gt_exists[
            np.arange(n_res).reshape((-1, 1)),
            atoms37_to_atoms14] * restype_atom14_mask
    restype_atom14_mask = restype_atom14_mask.reshape(n_res, 14, 1)
    atom_positions = atom_positions * restype_atom14_mask

    return atom_positions, restype_atom14_mask

def _backbone_frame_debug(
        backbone_frame: np.ndarray,
        debug: bool = False,
) -> None:
    if not debug:
        return
    num_res = backbone_frame.shape[0]
    assert np.allclose(backbone_frame[:, 1, :], np.zeros((num_res, 3)))
    assert np.allclose(backbone_frame[:, 2, 1:], np.zeros((num_res, 2)))
    assert np.allclose(backbone_frame[:, 0, 2:], np.zeros((num_res, 1)))
    assert (backbone_frame[:, 2, 0] > 0).all(), 'C in the backbone local frame must be positive x'
    assert (backbone_frame[:, 0, 1] > 0).all(), 'N in the backbone local frame must be positive y'

def extract_backbone_template(
        backbone_pos: np.ndarray,
) -> Tuple[np.ndarray, ...]:

    num_res = backbone_pos.shape[0]
    pos_dim = backbone_pos.shape[1]
    assert pos_dim in [4, 5], \
        'Expected input args `backbone_pos` shape ' \
        f'in [N, 4] (wo CB) or [N, 5] (w CB), but got shape {backbone_pos.shape}'
    include_cb = (pos_dim == 5)

    # records
    template_rigid_group = np.zeros([num_res, pos_dim, 3], dtype = np.float32)  # rigid group template
    restype_frame = np.zeros([num_res, 4, 4, 4], dtype = np.float32)  # default frame

    # 1. get backbone frame
    N_pos = backbone_pos[:, 0]
    CA_pos = backbone_pos[:, 1]
    C_pos = backbone_pos[:, 2]
    rots, translation = residue_frame(
        CA_pos, C_pos, N_pos)
    backbone_frame = apply_inv_euclidean(
        backbone_pos, rots, translation)

    _backbone_frame_debug(backbone_frame, False)

    template_rigid_group[:, 0, :2] = backbone_frame[:, 0, :2]  # N
    template_rigid_group[:, 2, :1] = backbone_frame[:, 2, :1]  # C
    if include_cb:
        # special case gly
        template_rigid_group[:, 4, :] = backbone_frame[:, 4, :]  # CB
    restype_frame[:, 0, :, :] = np.eye(4)

    # 2. pre-omega-group is empty
    restype_frame[:, 1, :, :] = np.eye(4)

    # 3. phi-group is empty, N->Ca as x-axis and ey (1, 0, 0) as xy-plane is R^T
    restype_frame[:, 2, :, :] = make_rigid_transformation_4x4(
        ex = template_rigid_group[:, 0] - template_rigid_group[:, 1],
        ey = np.tile(np.array([1.0, 0.0, 0.0]), (num_res, 1)),
        translation = template_rigid_group[:, 0, :])

    # 4. psi-frame to backbone, R^T = R
    # R = array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    restype_frame[:, 3, :, :] = make_rigid_transformation_4x4(
        ex = template_rigid_group[:, 2] - template_rigid_group[:, 1],
        ey = template_rigid_group[:, 1] - template_rigid_group[:, 0],
        translation = template_rigid_group[:, 2])
    ## extract the O atom pos in the local frame of psi-frame
    psi_frame = apply_inv_euclidean(
        backbone_frame, restype_frame[:, 3, :3, :3],
        template_rigid_group[:, 2])
    ### make sure O pos has positive y
    O_pos_over_psi_frame, psi_radian = parse_xrot_angle(psi_frame[:, 3])
    template_rigid_group[:, 3] = O_pos_over_psi_frame  # O

    return template_rigid_group, restype_frame, psi_radian.reshape(-1, 1)

def extract_chi_and_template(
        aatype: np.ndarray,
        atom_positions: np.ndarray,
        restype_atom14_mask: np.ndarray,
        return_radian: bool = False,
) -> ed:
    """
    Extract 3D structure translation and rotation, just like af2's implementation,
    Extract chi angles, rigid alphafold2 template repr, and default frame to transform
        the template to C-CA-N local frame following alphafold2 template definition.
    Returns:
        EasyDict object:
         atom14_position: Shape [N, 14, 3]
         sequence: Shape [N,]
         backbone_transl: Shape [N, 3]
         backbone_rots: Shape [N, 3, 3]
         default_frame: Shape [N, 8, 4, 4]
         rigid_group_positions: Shape [N, 14, 3]
         torsion_angle: Shape [N, 5, 2] or [N, 5]
    """
    num_res = aatype.shape[0]
    chi_angles_to_atoms14 = pc.chi_angles_to_atoms14_mapper[aatype]
    chi_angles_mask = np.array(pc.chi_angles_mask)[aatype]
    restype_atom14_to_rigid_group = pc.restype_atom14_to_rigid_group[aatype]

    # records
    template_rigid_group = np.zeros([num_res, 14, 3], dtype = np.float32)  # rigid group template
    restype_frame = np.zeros([num_res, 8, 4, 4], dtype = np.float32)  # default frame
    angle_radian = np.zeros([num_res, 5], dtype = np.float32)  # psi, chi 1-4

    # 1. get backbone frame
    N_pos = atom_positions[:, 0]
    CA_pos = atom_positions[:, 1]
    C_pos = atom_positions[:, 2]
    rots, translation = residue_frame(
        CA_pos, C_pos, N_pos)
    backbone_frame = apply_inv_euclidean(
        atom_positions, rots, translation)

    _backbone_frame_debug(backbone_frame, False)

    template_rigid_group[:, 0, :2] = backbone_frame[:, 0, :2] # N
    template_rigid_group[:, 2, :1] = backbone_frame[:, 2, :1] # C
    # special case gly
    template_rigid_group[:, 4, :] = backbone_frame[:, 4, :] # CB
    restype_frame[:, 0, :, :] = np.eye(4)

    # 2. pre-omega-group is empty
    restype_frame[:, 1, :, :] = np.eye(4)

    # 3. phi-group is empty, N->Ca as x-axis and ey (1, 0, 0) as xy-plane is R^T
    restype_frame[:, 2, :, :] = make_rigid_transformation_4x4(
        ex = template_rigid_group[:, 0] - template_rigid_group[:, 1],
        ey = np.tile(np.array([1.0, 0.0, 0.0]), (num_res, 1)),
        translation = template_rigid_group[:, 0, :])

    # 4. psi-frame to backbone, R^T = R
    # R = array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    restype_frame[:, 3, :, :] = make_rigid_transformation_4x4(
        ex = template_rigid_group[:, 2] - template_rigid_group[:, 1],
        ey = template_rigid_group[:, 1] - template_rigid_group[:, 0],
        translation = template_rigid_group[:, 2])
    ## extract the O atom pos in the local frame of psi-frame
    psi_frame = apply_inv_euclidean(
        backbone_frame, restype_frame[:, 3, :3, :3],
        template_rigid_group[:, 2])
    ### make sure O pos has positive y
    O_pos_over_psi_frame, psi_radian = parse_xrot_angle(psi_frame[:, 3])
    template_rigid_group[:, 3] = O_pos_over_psi_frame # O
    angle_radian[:, 0] = psi_radian

    # 5. chi(1,2,3,4)-frame to frame_before, when chi1, before-frame is backbone
    chi_before_frame = backbone_frame
    for _ in range(4):
        _mask = chi_angles_mask[:, _]
        if np.sum(_mask) == 0:
            continue
        _mask = _mask.astype('bool')
        subset_pos14 = chi_before_frame[_mask]
        n_sub = subset_pos14.shape[0]
        chi_mapper = chi_angles_to_atoms14[_mask, _]
        subset_pos4 = subset_pos14[np.arange(n_sub).reshape(-1, 1), chi_mapper]
        if _ == 0:
            mat = make_rigid_transformation_4x4(
                ex = subset_pos4[:, 2] - subset_pos4[:, 1],
                ey = subset_pos4[:, 0] - subset_pos4[:, 1],
                translation = subset_pos4[:, 2])
        else:
            _ey = np.tile(np.array([-1.0, 0.0, 0.0], dtype = np.float32), (n_sub, 1))
            mat = make_rigid_transformation_4x4(
                ex = subset_pos4[:, 2], ey = _ey,
                translation = subset_pos4[:, 2])
        # parse chi angles
        restype_frame[_mask, 4 + _, :, :] = mat
        subset_pos14_local_frame = apply_inv_euclidean(
            subset_pos14, mat[:, :3, :3], subset_pos4[:, 2])
        _next_frame_pos4 = subset_pos14_local_frame[np.arange(n_sub).reshape(-1, 1), chi_mapper]
        last_pos_over_frame, chi_radian = parse_xrot_angle(_next_frame_pos4[:, 3])
        angle_radian[_mask, _ + 1] = chi_radian
        # rotate the pos in the current frame, make sure the last atom from dihedral atoms is on xy-plane and write template
        subset_pos14_local_frame_rotx = rot_vec_around_x_axis(
            subset_pos14_local_frame, -chi_radian)
        _group_mask = (restype_atom14_to_rigid_group[_mask] == (_ + 4))
        _rigid_group_mask14 = np.where(_group_mask)
        _sum = np.zeros((n_sub, 14, 3), dtype = np.float32)
        _sum[_rigid_group_mask14] = subset_pos14_local_frame_rotx[_rigid_group_mask14]
        template_rigid_group[_mask] = template_rigid_group[_mask] + _sum

        # debug for side chain missing protein
        chi_mapper14_last = chi_angles_to_atoms14[_mask, _, 3]
        assert np.allclose(template_rigid_group[_mask.nonzero()[0], chi_mapper14_last], last_pos_over_frame), \
            'This error is likely because the pdb file has missing side chain atoms. Fixing the protein maybe fix the problem.'
        # update the residue pos to current local frame
        chi_before_frame[_mask] = subset_pos14_local_frame_rotx

    return ed(
        {
            'atom14_position': atom_positions,
             'sequence': aatype,
             'backbone_transl': translation,
             'backbone_rots': rots,
             'default_frame': restype_frame,
             'rigid_group_positions': template_rigid_group * restype_atom14_mask,
             'torsion_angle': angle_radian if return_radian else radian2sincos(angle_radian),
         }
    )

def build_pdb_from_template(
        template: ed,
        device: torch.device,
) -> Tuple[Tensor, ...]:
    """
    Used alphafold2 rule defined template with chi angle
        and global transl and rot to build a protein model.
    Args:
        template: EasyDict object, See more details from
            :method:`Protein.extract_chi_and_template`
            must include:
            sequence: Shape [N,]
            backbone_transl: Shape [N, 3]
            backbone_rots: Shape [N, 3, 3]
            default_frame: Shape [N, 8, 4, 4] or None
            rigid_group_positions: Shape [N, 14, 3] or None
            torsion_angle: Shape [N, 5, 2] or [N, 7, 2]
    Returns:
        pos14 repre with Shape [N, 14, 3] and mask14 with Shape [N, 14, 1]
    """
    template = ed(
        tree_map(
            lambda x: torch.from_numpy(x).float().to(device),
            template,
            np.ndarray
    ))
    template.sequence = template.sequence.long()
    mask = torch.ones(
        template.backbone_transl.shape[0],
        dtype = torch.bool, device = device)
    backbone_frames = aaframe.AAFrame(
        translation = template.backbone_transl,
        rotation = template.backbone_rots,
        unit = 'Angstrom',
        mask = mask)
    torsion_angles_mask = torch.ones_like(
        template.torsion_angle[..., 0],
        dtype = torch.bool)
    frames8 = backbone_frames.expand_w_torsion(
        torsion_angles = template.torsion_angle,
        torsion_angles_mask = torsion_angles_mask,
        fasta = template.sequence,
        rigid_frame = template.default_frame)
    pos14, mask14 = frames8.expanded_to_pos(
        template.sequence,
        template_pos = template.rigid_group_positions)
    mask14 = mask14.unsqueeze(-1)

    return pos14, mask14


def atom14_to_atom37(
        atom14_pos: Tensor,
        residx_atom37_to_atom14: Tensor,
        atom37_atom_exists: Tensor,
) -> Tensor:
    """
    Convert atom14_pos repr to atom37_pos by using
        residx_atom37_to_atom14 operator and using
        atom37_atom_exists to mask wrong pos;
    Args:
        atom14_pos: (*, N, 14, 3).
        residx_atom37_to_atom14: (*, N, 37).
        atom37_atom_exists: (*, N, 37).
    Returns:
        atom37_pos: (*, N, 37, 3)
    """
    atom37_pos = batched_gather(
        atom14_pos,
        residx_atom37_to_atom14,
        dim = -2,
        batch_ndims = len(atom14_pos.shape[:-2])
    )
    return atom37_pos * atom37_atom_exists[..., None]


def get_chi_atom_indices(repr_int: int = 37):
    """
    Returns atom indices needed to compute chi angles for all residue types.
    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    assert repr_int in [14, 37], 'Only support atom14 or atom37 repr'
    chi_atom_indices = []
    for residue_name in pc.restypes:
        residue_name = pc.restype_1to3[residue_name]
        residue_chi_angles = pc.chi_angles_atoms[residue_name]
        atom_indices = []
        if repr_int == 37:
            atom_index = lambda x:pc.atom_order[x]
        else:
            atom_index = lambda x:pc.restype_name_to_atom14_names[residue_name].index(x)
        for chi_angle in residue_chi_angles:
            atom_indices.append([atom_index(atom) for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append(
                [0, 0, 0, 0]
            )  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return chi_atom_indices

def make_torsion_mask(
        aatype: Tensor,
        all_atom_mask: Tensor,
) -> Tensor:
    """
    Args:
        aatype: shape (*, N);
        all_atom_mask: shape (*, N, 37/14)
    Returns:
        chi_mask: BoolTensor with shape (*, N, 4) indicating whether residues
            have no any mising chi atoms for each should-existing torsion angle.
    """
    atom_repr_type = all_atom_mask.shape[-1]
    all_atom_mask = all_atom_mask.float()
    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices(atom_repr_type), device = aatype.device
    )
    # (*, N, 4, 4)
    atom_indices = chi_atom_indices[..., aatype, :, :]
    chi_angles_mask = list(pc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)
    # (*, N, 4)
    chis_mask = chi_angles_mask[aatype, :]
    # (*, N, 4, 4)
    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim = -1,
        batch_ndims = len(atom_indices.shape[:-2]),
    )
    # (*, N, 4)
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim = -1,
        dtype = chi_angle_atoms_mask.dtype,
    )
    chis_mask = (chis_mask * chi_angle_atoms_mask).bool()
    # # (*, N)
    # torsion_mask = (chis_mask == chi_angle_atoms_mask)
    # torsion_mask = torsion_mask.all(dim = -1)

    return chis_mask