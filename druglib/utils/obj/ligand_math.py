# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Optional, Union,
    Tuple, List,
)
import numpy as np
import torch
from torch import Tensor

from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import GetUFFVdWParams

from . import ligand_constants as lc
from ..torch_utils import (
    maybe_num_nodes, to_dense_adj,
)


def merge_edge(
        extra_edge: np.ndarray,
        edge_index: np.ndarray,
        bond_type: np.ndarray,
        num_nodes: Optional[int] = None,
):
    """
    Helper function: Add extra edge into the `edge_index`.
    Merge the original chemical bond `edge_index` with the input `extra_edge`，
        also called geometry graph connectivity.
    Args:
        extra_edge: np.ndarray. Shape (2, num_edges)
        Note that must be from the same :obj:Chem.rdchem.Mol
        and the atom idx must be the same between the edge_index
        and `extra_edge`.
    Returns:
        new edge_index and edge_attr (using connect_types idx)
        new_add_mask: Indicate which edges are the new compared with
            original `edge_index`.
    E.g.:
        >>> edge_index = np.array([[1, 1, 2, 3], [2, 3, 4, 2]])# self.edge_index
        >>> N = 5# self.numatoms
        >>> bond_type = np.array([1, 0, 2, 0])# self.bond_type
        >>> extra_edge = np.array([[0, 4, 1, 1], [1, 0, 2, 3]])
        >>> # merge_edge(..., extra_edge)# lc.connect_to_id['NoneType'] -> 5
        (array([[0, 1, 1, 2, 3, 4],
                [1, 2, 3, 4, 2, 0]]),
         array([5,  1,  0,  2,  0, 5]),
         array([1,  0,  0,  0,  0, 1]))
    """
    if num_nodes is None:
       num_nodes = maybe_num_nodes(torch.from_numpy(edge_index), num_nodes)

    _edge_index = np.concatenate([edge_index, extra_edge], axis = 1)
    _edge_index = np.unique(_edge_index, axis = 1)
    numatoms_sqr = num_nodes ** 2
    chem_graph_id_row = (edge_index[0] * num_nodes + edge_index[1]).astype(int)
    new_graph_id_row = (_edge_index[0] * num_nodes + _edge_index[1]).astype(int)

    flatten = np.full(
        numatoms_sqr,
        fill_value = -1,
        dtype = int,
    )
    flatten[new_graph_id_row] = lc.connect_to_id['NoneType']
    flatten[chem_graph_id_row] = bond_type
    bond_type = flatten[new_graph_id_row]
    new_edge_mask = np.zeros(
        numatoms_sqr,
        dtype = int,
    )
    new_edge_mask[new_graph_id_row] = 1
    new_edge_mask[chem_graph_id_row] = 0
    row, col = (flatten.reshape(num_nodes, num_nodes) > -0.5).nonzero()

    return np.stack([row, col], axis = 0), bond_type, new_edge_mask[new_graph_id_row]

def vdw_radius(
        mol: Chem.rdchem.Mol
) -> np.ndarray:
    """Get vdW radius for every atom."""
    vdwr = []
    for idx, at in enumerate(mol.GetAtoms()):
        sym = at.GetSymbol().capitalize()
        if sym in lc.vdW_radius:
            _r = lc.vdW_radius[sym]
        else:
            _r = lc.pt.GetRvdw(sym)
        vdwr.append(float(_r))

    return np.array(vdwr, dtype = float)

def cov_adj(
        atom1typeid: int,
        atom2typeid: int,
        bondtypeid: int,
) -> float:
    """
    Get covalent length from bond type and connected atoms type.
    Bond type id from druglib ligand parameters and atom id from
        periodic table.
    """
    if bondtypeid in [lc.bondtypes_to_id[bt] for bt in ['SINGLE', 'other']]:
        # `other` bond type is always single-bond type
        bl_dict = lc.single_bond_length_pid
        covbr = lc.single_covbr_pid
    elif bondtypeid in [lc.bondtypes_to_id[bt] for bt in ['DOUBLE', 'AROMATIC']]:
        # use double bond type as clash penalty for AROMATIC
        bl_dict = lc.double_bond_length_pid
        covbr = lc.double_covbr_pid
    elif bondtypeid == lc.bondtypes_to_id['TRIPLE']:
        bl_dict = lc.triplet_bond_length_pid
        covbr = lc.triple_covbr_pid
    else:
        raise ValueError(f'bondtupeid {bondtypeid} not found in bondtypes library: {lc.bondtypes_to_id}')

    if (atom1typeid in bl_dict) and (atom2typeid in bl_dict[atom1typeid]):
        length = bl_dict[atom1typeid][atom2typeid]
    elif (atom1typeid in covbr) and (atom2typeid in covbr):
        length = covbr[atom1typeid] + covbr[atom2typeid]
    else:
        length = sum([lc.pt.GetRcovalent(at) for at in [atom1typeid, atom2typeid]])

    return length

def make_cov_tensor(
        atomtype: Union[np.ndarray, Tensor],
        edge_index: Union[np.ndarray, Tensor],
        bond_type: Union[np.ndarray, Tensor],
) -> Tensor:
    """
    Make covalent bond distance lower bound.
    Args:
        atomtype: Tensor or np.ndarray with shape (N,), indices from periodic table.
        edge_index: Tensor or np.ndarray with shape (2, N_edges),
            with N_edges covalent edges.
        bond_type: Tensor or np.ndarray with shape (N_edges)
    Returns:
        covalent bond distance lower bound with shape (N_edges,), FloatTensor.
    """
    covalent_lower_bound = []
    for idx, edge in enumerate(edge_index.T):
        i, j = edge
        at1 = int(atomtype[i])
        at2 = int(atomtype[j])
        bt = int(bond_type[idx])
        covalent_lower_bound.append(cov_adj(at1, at2, bt))

    return torch.tensor(covalent_lower_bound).float()

def uff_vdw_param(
        mol: Chem.rdchem.Mol,
        atom1: int,
        atom2: int,
) -> Optional[Tuple[float, float]]:
    """Get UFF vdW parameters"""
    param = GetUFFVdWParams(mol, atom1, atom2)
    # if the parameters is None, using atom
    if param is None:
        # warnings.warn(
        #     f'UFF parameters is None between atom1 {atom1} and atom2 {atom2}. '
        #     'Here we set param to 0, 0, so we can make mask.'
        # )
        return 0, 0
    rij, epsilon = param
    return rij, epsilon

def make_vdw_param(
        mol: Chem.rdchem.Mol,
        edge_index: Union[np.ndarray, Tensor],
        num_nodes: Optional[int] = None,
) -> Tuple[Tensor, ...]:
    """
    Make van der Waals parameters.
    Args:
        mol: rdkit.Chem.rdchem.Mol.
        edge_index: Tensor or np.ndarray with shape (2, N_edges),
            with N_edges covalent edges.
        num_nodes: int, optional. The number of atoms.
    Returns:
        Tuple of Tensor: nonbond_edges: LongTensor with shape (2, M) undirected graph.
        uff_param: FloatTensor with shape (2, M) with row 0 is epsilon
            and row 1 is rij parameters.
    """
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index).long()
    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index)
    adj = to_dense_adj(edge_index, max_num_nodes = num_nodes).squeeze(0)

    nonbond_edges, uff_param = [], []
    for i in range(num_nodes):
        for j in range(i):
            if adj[i, j] < 0.5:
                nonbond_edges.append([i, j])
                _rij, _epsilon = uff_vdw_param(mol, i, j)
                uff_param.append([_epsilon, _rij])

    nonbond_edges = torch.tensor(nonbond_edges).long().T
    uff_param = torch.tensor(uff_param).float().T

    return nonbond_edges, uff_param

def make_angle_indices(
        mol: Chem.rdchem.Mol,
        ignore_ptid: List[int] = [1,],
) -> Tuple[Tensor, ...]:
    """
    Make Mol angle indices for angle supervised loss.
    Note: We ignore the i->j<-k where i or k is hydrogen atom or other elements.
    Args:
        mol: rdkit.Chem.rdchem.Mol.
        ignore_ptid: The atom idx from eriodic table to be ingored.
    Returns:
        ang_src_index: LongTensor with shape (1, N_center), angle centre atoms idx。
        ang_dst_index: LongTensor with shape (6, N_center), maximal 6 neighborhoods.
        ang_mask: BoolTensor with shape (6, N_center), True indicating the neighborhood atoms exists
    """
    ignore_ptid = [lc.pt.GetElementSymbol(atid) for atid in ignore_ptid]
    ang_src, ang_dst = [], []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ignore_ptid:
            continue
        n_ids = [
            n.GetIdx() for n in atom.GetNeighbors()\
                if n.GetSymbol() not in ignore_ptid
        ]
        if len(n_ids) > 1:
            ang_src.append(atom.GetIdx())
            ang_dst.append(n_ids)
    num_neighs = len(ang_src)
    ang_src_index = torch.tensor(ang_src, dtype = torch.long).unsqueeze(0)
    ang_dst_index = torch.zeros((6, num_neighs), dtype = torch.long)
    ang_mask = torch.zeros((6, num_neighs), dtype = torch.bool)

    for i, n_ids in enumerate(ang_dst):
        # ignore abnormal bond angle
        if len(n_ids) > 6:
            continue
        ang_dst_index[: len(n_ids), i] = torch.LongTensor(n_ids)
        ang_mask[: len(n_ids), i] = True

    return ang_src_index, ang_dst_index, ang_mask