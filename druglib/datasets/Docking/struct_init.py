# Copyright (c) MDLDrugLib. All rights reserved.
import logging
from easydict import EasyDict as ed
import numpy as np
from scipy.spatial.transform import Rotation
import torch

import druglib.utils.obj.protein_constants as pc
from druglib.utils.obj import build_pdb_from_template
from druglib.utils.geometry_utils import radian2sincos_torch
from druglib.utils.bio_utils import modify_conformer_torsion_angles
from ..builder import PIPELINES


@PIPELINES.register_module()
class LigInit:
    """Ligand position initialization from priori distribution"""
    def __init__(
            self,
            tr_sigma_max: float = 10.,
    ):
        self.tr_sigma_max = tr_sigma_max

    def __call__(self, data):
        lig_pose = data['lig_pos']

        num_torsion_bonds = data['tor_edge_mask'].sum()
        no_torsion = (num_torsion_bonds == 0)
        if not no_torsion:
            torsion_updates = np.random.uniform(
                low = -np.pi, high = np.pi,
                size = num_torsion_bonds,
            )
            rot_node_mask = data['metastore']['rot_node_mask']
            lig_pose = modify_conformer_torsion_angles(
                lig_pose,
                data['lig_edge_index'].T[data['tor_edge_mask'].bool()],
                rot_node_mask,
                torsion_updates)

        center = torch.mean(lig_pose, dim = 0, keepdim = True)
        random_rotation = torch.from_numpy(
            Rotation.random().as_matrix(),
        ).float()
        tr_update = torch.normal(
            mean = 0,
            std = self.tr_sigma_max,
            size = (1, 3),
        )
        lig_pose = (lig_pose - center) @ random_rotation.T + tr_update
        data['lig_pos'] = lig_pose

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'tr_sigma_max={self.tr_sigma_max}'
                f')'
                )

@PIPELINES.register_module()
class SCFixer:
    """
    Side chain fix if some chi atoms are missing.
    """
    def __init__(self, strict: bool = False):
        from druglib.utils import get_logger
        self.report = get_logger('Side-Chain-Missing-Reporter')
        if strict:
            self.report.setLevel(logging.DEBUG)

    def __call__(self, data):
        sequence = data['sequence']
        sc_torsion_edge_mask = data['sc_torsion_edge_mask']
        torsion_exists_mask = np.asarray(pc.chi_angles_mask)[sequence]
        torsion_exists_mask = torch.from_numpy(torsion_exists_mask).bool()
        torsion_mask = (torsion_exists_mask == sc_torsion_edge_mask)
        torsion_mask = torsion_mask.all(dim = -1)
        not_mask = torch.logical_not(torsion_mask)
        num_missing = not_mask.sum()
        if num_missing == 0:
            return data
        # avoid residue frame missing and chi atoms exists
        atom14_mask = data['atom14_mask']
        bb_exits = atom14_mask[:, :3].bool().all()
        num_bb_missing = torch.logical_not(bb_exits).sum()
        self.report.debug(
            f'Side chain missing number: {num_missing}, '
            f'while backbone frame missing number {num_bb_missing}.')
        not_mask = torch.logical_and(not_mask, bb_exits)
        sc_torsion_edge_mask[not_mask] = torsion_exists_mask[not_mask]
        data['sc_torsion_edge_mask'] = sc_torsion_edge_mask * bb_exits.unsqueeze(-1)

        # reset default frame and rigid_group_positions to af2 template
        residues = sequence[not_mask]
        default_frame = data['default_frame']
        rigid_group_positions = data['rigid_group_positions']
        default_frame[not_mask] = torch.from_numpy(pc.restype_rigid_group_default_frame).float()[residues]
        rigid_group_positions[not_mask] = torch.from_numpy(pc.restype_atom14_rigid_group_positions).float()[residues]
        atom14_mask[not_mask] = torch.from_numpy(pc.restype_atom14_mask).bool()[residues]
        data['default_frame'] = default_frame
        data['rigid_group_positions'] = rigid_group_positions
        data['atom14_mask'] = atom14_mask

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f')'
                )

@PIPELINES.register_module()
class SCProtInit:
    """Given fixed protein backbone, randomly initialize the side chain"""
    def __call__(self, data):
        n_res = data['sequence'].shape[0]
        sc_torsion_edge_mask = data['sc_torsion_edge_mask']
        torsion_updates = np.random.uniform(
            low = -np.pi, high = np.pi,
            size = (n_res, 4))
        torsion_angle = data['torsion_angle']
        assert torsion_angle.shape == (n_res, 5), 'torsion angle should be (N, 5)'
        torsion_angle[:, 1:] = torch.from_numpy(torsion_updates * sc_torsion_edge_mask.numpy())
        data['torsion_angle'] = torsion_angle
        pos14, mask14 = build_pdb_from_template(
            ed(sequence = data['sequence'],
               backbone_transl = data['backbone_transl'],
               backbone_rots = data['backbone_rots'],
               default_frame = data['default_frame'],
               rigid_group_positions = data['rigid_group_positions'],
               torsion_angle = radian2sincos_torch(torsion_angle),
               ), torch.device('cpu'))
        pos14 = pos14 * data['atom14_mask'].unsqueeze(-1)
        data['atom14_position'] = pos14

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f')'
                )
