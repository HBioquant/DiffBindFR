# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Tuple
import numpy as np
import networkx as nx

import torch
from torch import Tensor, BoolTensor
from druglib.utils.obj import Ligand3D, make_torsion_mask
from druglib.utils.obj import protein_constants as pc
from druglib.utils.torch_utils import to_nx, batched_gather


def nx_from_Ligand3D(
        ligand: Ligand3D,
        add_edge_match: bool = False
) -> nx.Graph:
    atom_prop = ligand.atom_prop
    node_attrs = np.stack(
        [
            ligand.atomtype,
            atom_prop['isaromatic'],
            atom_prop['formal_charge'],
            atom_prop['degree'],
        ], axis = -1,
    )

    bond_prop = ligand.bond_prop
    bond_label = bond_prop['bond_label']
    mask = (bond_label == 0)
    edge_index = ligand.edge_index[:, mask]
    edge_attrs = None
    if add_edge_match:
        edge_attrs = np.stack(
            [
                bond_prop['isinring'][mask],
                bond_prop['isconjugated'][mask]
            ], axis = -1,
        )
    nxg = to_nx(
        node_attrs = node_attrs,
        edge_index = edge_index,
        edge_attrs = edge_attrs,
    )

    return nxg

def find_torsion(
        ligand: Ligand3D,
) -> Tuple[np.ndarray, ...]:
    """Find rotatable bonds of ligand"""
    bond_label = ligand.bond_prop['bond_label']
    covalent_edge_mask = (bond_label == 0)
    edge_index = ligand.edge_index
    covalent_edge = edge_index[:, covalent_edge_mask]

    # convert to networkx
    G = nx.DiGraph()
    assert ligand.numatoms == ligand.atomtype.shape[0]
    G.add_nodes_from(range(ligand.numatoms))
    for i, (u, v) in enumerate(covalent_edge.T.tolist()):
        G.add_edge(u, v)

    assert nx.is_connected(G.to_undirected()), 'Find ligand graph is already disconnecting.'
    
    # torsion bond definition
    _tor_edge_mask = []
    rot_node_mask = []
    edges_T = covalent_edge.T
    for i in range(edges_T.shape[0]):
        G_undirected = G.to_undirected()
        G_undirected.remove_edge(*edges_T[i])
        if not nx.is_connected(G_undirected):
            small_frag = list(sorted(
                nx.connected_components(G_undirected),
                key = len)[0])
            if (len(small_frag) > 1) and (edges_T[i, 1] in small_frag):
                _tor_edge_mask.append(1)
                _rot_mask = np.zeros(G.number_of_nodes(), dtype = bool)
                _rot_mask[np.asarray(small_frag, dtype = int)] = True
                rot_node_mask.append(_rot_mask)
                continue
        _tor_edge_mask.append(0)

    _tor_edge_mask = np.asarray(_tor_edge_mask, dtype = bool)
    tor_edge_mask = np.zeros_like(covalent_edge_mask, dtype = bool)
    tor_edge_mask[covalent_edge_mask] = _tor_edge_mask

    rot_node_mask = np.asarray(rot_node_mask, dtype = bool)
    if len(rot_node_mask) == 0:
        rot_node_mask = np.empty((0, G.number_of_nodes()), dtype = bool)

    return tor_edge_mask, rot_node_mask

def build_torsion_edges(
        sequence: Tensor,
        atom14_mask: BoolTensor,
) -> Tuple[Tensor, ...]:
    """
    Args:
        sequence: Tensor. Shape (N_res, )
        atom14_mask: BoolTensor. Shape (N_res, 14)
    Returns:
        edge_index: LongTensor. i->j->k<-l shape (N_res, 4, 3, 2)
        torsion_mask: BoolTensor with shape (N_res) indicating whether residues
            have no any mising chi atoms.
        chi_mask: BoolTensor with shape (N_res, 4) indicating whether residues
            have no any mising chi atoms for each should-existing torsion angle.
     """
    if not isinstance(atom14_mask, BoolTensor):
        atom14_mask = atom14_mask.bool()
    node_idx = torch.zeros(
        atom14_mask.shape, dtype = torch.long)
    node_idx[atom14_mask] = torch.arange(
        atom14_mask.sum(), dtype = torch.long)
    atom14_torsion_edges = pc.restype_atom14_torsion_edges[sequence]
    atom14_torsion_edges = torch.from_numpy(atom14_torsion_edges).long()
    atom14_torsion_edges = batched_gather(
        node_idx,
        atom14_torsion_edges,
        -1, len(node_idx.shape[:-1])
    )
    # (N_res, 4)
    chis_mask = make_torsion_mask(
        sequence, atom14_mask
    )
    atom14_torsion_edges = atom14_torsion_edges * chis_mask[..., None, None]
    return atom14_torsion_edges, chis_mask