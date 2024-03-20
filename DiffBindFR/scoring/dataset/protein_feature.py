# Copyright (c) MDLDrugLib. All rights reserved.
import math
from tqdm import tqdm
tqdm.pandas()
import numpy as np
from scipy.spatial import distance_matrix

import torch
import torch_cluster
import torch.nn.functional as F
from openfold.np import residue_constants, protein
from openfold.data.data_transforms import (
    make_atom14_masks,
    make_atom14_positions,
    atom37_to_torsion_angles,
    get_backbone_frames,
    squeeze_features,
    make_seq_mask,
)


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def get_orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def get_sidechains(n, ca, c):
    c, n = _normalize(c - ca), _normalize(n - ca)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec

def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]]
        for i in range(len(aatype))
    ])


def make_sequence_features(
        sequence: str, description: str, num_res: int
):
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features


def make_protein_features(
        protein_object: protein.Protein,
        description: str,
        _is_distillation: bool = False,
):
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(
        1. if _is_distillation else 0.
    ).astype(np.float32)

    return pdb_feats


def make_pdb_features(
        protein_object: protein.Protein,
        description: str,
        is_distillation: bool = True,
        confidence_threshold: float = 50.,
):
    pdb_feats = make_protein_features(
        protein_object, description, _is_distillation=True
    )

    if (is_distillation):
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats


def get_protein_feature(
        pocket_pdb_path: str,
        topk: int = 30,
        pdb_string: bool = False,
):
    batch = {}
    if not pdb_string:
        with open(pocket_pdb_path, 'r') as f:
            pdb_str = f.read()
    else:
        pdb_str = pocket_pdb_path
    protein_object = protein.from_pdb_string(pdb_str, None)
    pdb_feats = make_pdb_features(
        protein_object,
        'try',
        is_distillation=False
    )
    pdb_feats.pop('domain_name')
    pdb_feats.pop('sequence')
    for key, var in pdb_feats.items():
        pdb_feats[key] = torch.from_numpy(var).unsqueeze(0)
    pdb_feats = squeeze_features(pdb_feats)
    pdb_feats = make_atom14_masks(pdb_feats)
    pdb_feats['aatype'] = pdb_feats['aatype'].long()
    pdb_feats['residue_index'] = pdb_feats['residue_index'].long()
    pdb_feats = make_atom14_positions(pdb_feats)
    pdb_feats = atom37_to_torsion_angles()(pdb_feats)
    pdb_feats = make_seq_mask(pdb_feats)

    het_mask_ = (pdb_feats['aatype'] != 20)

    batch['atom14_position'] = pdb_feats['atom14_gt_positions'][het_mask_]
    batch['aatype'] = pdb_feats['aatype'][het_mask_]
    batch['atom14_mask'] = pdb_feats['atom14_atom_exists'][het_mask_].to(torch.int64)

    batch['all_torsion_angles_sin_cos'] = pdb_feats['torsion_angles_sin_cos'][het_mask_]
    batch['all_alt_torsion_angles_sin_cos'] = pdb_feats['alt_torsion_angles_sin_cos'][het_mask_]
    batch['all_torsion_angles_mask'] = pdb_feats['torsion_angles_mask'][het_mask_]

    bb_mask = (batch['atom14_mask'][:, 0] * batch['atom14_mask'][:, 1] *
               batch['atom14_mask'][:, 2] * batch['atom14_mask'][:, 3]).bool()

    for k, v in batch.items():
        batch[k] = v[bb_mask]

    intra_dis = [
        0.1 * torch.linalg.norm((batch['atom14_position'][:, 1] - batch['atom14_position'][:, 3]) + 1e-6, dim=-1),
        0.1 * torch.linalg.norm((batch['atom14_position'][:, 0] - batch['atom14_position'][:, 3]) + 1e-6, dim=-1),
        0.1 * torch.linalg.norm((batch['atom14_position'][:, 0] - batch['atom14_position'][:, 2]) + 1e-6, dim=-1),
        ]
    seq = batch['aatype']

    bb_dihedral_rad = batch['all_torsion_angles_sin_cos'][:, :3].view(-1, 6)  # waiting for re-make the features
    node_s = torch.cat([torch.stack(intra_dis).T, bb_dihedral_rad], -1)
    # edge features
    X_center_of_mass = batch['atom14_position'].sum(-2) / batch['atom14_mask'].sum(-1)[:, None]
    edge_index = torch_cluster.knn_graph(batch['atom14_position'][:, 1], k=topk)
    dis_minmax = torch.stack([0.1 * torch.linalg.norm(
        (batch['atom14_position'][edge_index[0], 1] - batch['atom14_position'][edge_index[1], 1]) + 1e-6, dim=-1),
                              0.1 * torch.linalg.norm((batch['atom14_position'][edge_index[0], 4] -
                                                       batch['atom14_position'][edge_index[1], 4]) + 1e-6, dim=-1)]).T
    dis_matx_center = distance_matrix(X_center_of_mass, X_center_of_mass)

    cadist = (torch.pairwise_distance(batch['atom14_position'][edge_index[0], 1],
                                      batch['atom14_position'][edge_index[1], 1]) * 0.1).view(-1, 1)
    cedist = (torch.from_numpy(dis_matx_center[edge_index[0, :], edge_index[1, :]]) * 0.1).view(-1, 1)  ## need to check
    edge_connect = (dis_minmax[:, 0] < 4.5).to(torch.float32).view(-1, 1)
    edge_s = torch.cat([edge_connect, cadist, cedist, dis_minmax, _rbf(dis_minmax[:, 0], D_count=16, device='cpu')],
                       dim=1)

    # vector features
    orientations = get_orientations(batch['atom14_position'][:, 1])
    sidechains = get_sidechains(n=batch['atom14_position'][:, 0], ca=batch['atom14_position'][:, 1],
                                c=batch['atom14_position'][:, 2])
    node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
    edge_v = _normalize(
        batch['atom14_position'][edge_index[0], 1] - batch['atom14_position'][edge_index[1], 1]).unsqueeze(-2)
    node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))

    return batch['atom14_position'][:, 1], batch['atom14_position'], \
           seq, node_s, node_v, edge_index, edge_s, edge_v