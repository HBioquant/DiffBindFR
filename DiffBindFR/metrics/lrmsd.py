# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Optional, Union,
    List, Tuple, Sequence,
    Mapping, Any
)
import warnings
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import numpy as np

import torch
from torch import Tensor

from rdkit import Chem
from rdkit.Chem import rdFMCS

from druglib import time_limit as tm_fn
from druglib.utils.torch_utils import batched_gather, match_graphs

possible_bond_type_list = ["AROMATIC", "TRIPLE", "DOUBLE", "SINGLE", "misc"]


def safe_index(l, e):
    try:
        return l.index(e) + 1
    except:
        return len(l)

def safe_index_bond(bond):
    return safe_index(possible_bond_type_list, str(bond.GetBondType()))

def atomGetnum(mol):
    atomnums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom2bonds = [0 for _ in range(len(atomnums))]
    if len(mol.GetBonds()) > 0:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_num = safe_index_bond(bond)
            atom2bonds[i] += bond_num
            atom2bonds[j] += bond_num

    # for i in range(len(atomnums)):
    #     atomnums[i] = atomnums[i] * 100 + atom2bonds[i]

    return atomnums, atom2bonds

def rdmol_to_nxgraph(rdmol):
    graph = nx.Graph()
    aprops, atom2bonds = atomGetnum(rdmol)

    for idx, atom in enumerate(rdmol.GetAtoms()):
        # Add the atoms as nodes
        graph.add_node(
            atom.GetIdx(),
            atom_type=atom.GetAtomicNum(),
            aprops=np.array([int(aprops[idx]), int(atom2bonds[idx])]),
        )

    # Add the bonds as edges
    for bond in rdmol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    return graph

def graph_from_adjacency_matrix(
        adjacency_matrix: Union[np.ndarray, List[List[int]]],
        aprops: Optional[Union[np.ndarray, List[Any]]] = None,
) -> nx.Graph:
    """
    Graph from adjacency matrix.
    Args:
        adjacency_matrix: Union[np.ndarray, List[List[int]]]
            Adjacency matrix
        aprops: Union[np.ndarray, List[Any]], optional
            Atomic properties
    Notes:
    It the atomic numbers are passed, they are used as node attributes.
    """
    G = nx.Graph(adjacency_matrix)

    if not nx.is_connected(G):
        warnings.warn('Disconnected graph detected.')

    if aprops is not None:
        attributes = {idx: aprops for idx, aprops in enumerate(aprops)}
        nx.set_node_attributes(G, attributes, "aprops")

    return G

def tree_compare(
        x, y,
) -> bool:
    assert type(x) == type(y), \
        f'x {type(x)} and y {type(y)} type mismatch'
    if isinstance(x, (np.ndarray, Tensor)):
        return (x == y).all()
    elif isinstance(x, Sequence) and not isinstance(x, str):
        return all([tree_compare(_x, _y) for _x, _y in zip(x, y)])
    elif isinstance(x, Mapping):
        assert len(x) == len(y), f'Mapping type x ({len(x)}) and y ({len(y)}) length mismatch'
        return all([tree_compare(x[k], y[k]) for k in x.keys()])
    else:
        return x == y

def match_aprops(
        node1,
        node2,
) -> bool:
    """
    Check if atomic properties for two nodes match.
    """
    node1_props = node1["aprops"]
    node2_props = node2["aprops"]
    return tree_compare(node1_props, node2_props)

def match_graphs_wo_em(
        G1,
        G2,
        keep_self: bool = False,
) -> List[Tuple[List[int], List[int]]]:
    if (
            nx.get_node_attributes(G1, "aprops") == {}
            or nx.get_node_attributes(G2, "aprops") == {}
    ):
        # Nodes without atomic number information
        # No node-matching check
        node_match = None

        warnings.warn('No atomic property information stored on nodes. Node matching is not performed...')

    else:
        # print('match_aprops function')
        node_match = match_aprops

    GM = nx.algorithms.isomorphism.GraphMatcher(G1, G2, node_match)

    # Check if graphs are actually isomorphic
    if not GM.is_isomorphic():
        raise ValueError('Graphs are not isomorphic.\nMake sure graphs have the same connectivity.')

    def sorted_array(
            seq1: List[int],
            seq2: List[int],
    ):
        """Sort seq1 by seq2 and deprecate seq1 because always \equiv np.arange(num_nodes)"""
        assert len(seq1) == len(seq2), f'seq1 ({len(seq1)}) and seq2 ({len(seq2)}) length mismatch'
        seq1 = np.array(seq1)
        seq2 = np.array(seq2)
        sort_ind = np.argsort(seq2)
        seq1 = seq1[sort_ind]
        seq2 = seq2[sort_ind]
        assert (seq2 == np.arange(seq2.shape[0])).all(), \
            f'Graph matching node missing {seq2} while {np.arange(seq2.shape[0])} are needed'
        return seq1

    isomorphisms = []
    for isomorphism in GM.isomorphisms_iter():
        isom_arr = sorted_array(list(isomorphism.keys()), list(isomorphism.values()))
        self = (isom_arr == np.arange(isom_arr.shape[0])).all()
        if not self:
            isomorphisms.append(isom_arr)
        elif keep_self:
            isomorphisms.append(isom_arr)

    return isomorphisms

def calc_rmsd_nx(mol_a, mol_b):
    """ Calculate RMSD of two molecules with unknown atom correspondence. """
    graph_a = rdmol_to_nxgraph(mol_a)
    graph_b = rdmol_to_nxgraph(mol_b)

    gm = GraphMatcher(
        graph_a, graph_b,
        node_match=match_aprops)

    isomorphisms = list(gm.isomorphisms_iter())
    isoms = match_graphs_wo_em(graph_a, graph_b)
    # print(isoms)
    if len(isomorphisms) < 1:
        return None

    all_rmsds = []
    for mapping in isomorphisms:
        atom_types_a = [atom.GetAtomicNum() for atom in mol_a.GetAtoms()]
        atom_types_b = [mol_b.GetAtomWithIdx(mapping[i]).GetAtomicNum()
                        for i in range(mol_b.GetNumAtoms())]
        assert atom_types_a == atom_types_b

        conf_a = mol_a.GetConformer()
        coords_a = np.array([conf_a.GetAtomPosition(i)
                             for i in range(mol_a.GetNumAtoms())])
        conf_b = mol_b.GetConformer()
        coords_b = np.array([conf_b.GetAtomPosition(mapping[i])
                             for i in range(mol_b.GetNumAtoms())])

        diff = coords_a - coords_b
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        all_rmsds.append(rmsd)

    if len(isomorphisms) > 1:
        print("More than one isomorphism found. Returning minimum RMSD.")

    return min(all_rmsds)

def get_symmetry_rmsd(mol, coords1, coords2, mol2=None):
    from spyrmsd import rmsd, molecule
    with tm_fn(10):
        mol = molecule.Molecule.from_rdkit(mol)
        mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        RMSD = rmsd.symmrmsd(
            coords1,
            coords2,
            mol.atomicnums,
            mol2_atomicnums,
            mol.adjacency_matrix,
            mol2_adjacency_matrix,
        )
        return RMSD

def CalcLigRMSD(
        lig1, lig2,
        rename_lig2 = False,
        output_filename = "tmp.pdb",
        removeHs = True,
):
    """
    Calculate the Root-mean-square deviation (RMSD) between two prealigned ligands,
        even when atom names between the two ligands are not matching.
    The symmetry of the molecules is taken into consideration (e.g. tri-methyl groups).
    Moreover, if one ligand structure has missing atoms (e.g. undefined electron density
        in the crystal structure), the RMSD is calculated for the maximum common substructure (MCS).
    Args:
        lig1 : RDKit molecule
        lig2 : RDKit molecule
        rename_lig2 : bool, optional
            True to rename the atoms of lig2 according to the atom names of lig1
        output_filename : str, optional
            If rename_lig2 is set to True, a PDB file with the renamed lig2 atoms is written as output.
            This may be useful to check that the RMSD has been "properly" calculated,
            i.e. that the atoms have been properly matched for the calculation of the RMSD.

    Returns：
        rmsd : float。 Root-mean-square deviation between the two input molecules
    """
    if removeHs:
        # Exclude hydrogen atoms from the RMSD calculation
        lig1 = Chem.RemoveHs(lig1)
        lig2 = Chem.RemoveHs(lig2)
    # Extract coordinates
    coordinates_lig2 = lig2.GetConformer().GetPositions()
    coordinates_lig1 = lig1.GetConformer().GetPositions()
    # Calculate the RMSD between the MCS of lig1 and lig2 (useful if e.g. the crystal structures has missing atoms)
    res = rdFMCS.FindMCS([lig1, lig2])
    ref_mol = Chem.MolFromSmarts(res.smartsString)
    # Match the ligands to the MCS
    # For lig2, the molecular symmetry is considered:
    # If 2 atoms are symmetric (3 and 4), two indeces combinations are printed out
    # ((0,1,2,3,4), (0,1,2,4,3)) and stored in mas2_list
    mas1 = list(lig1.GetSubstructMatch(ref_mol))  # match lig1 to MCS
    mas2_list = lig2.GetSubstructMatches(ref_mol, uniquify=False)
    # Reorder the coordinates of the ligands and calculate the RMSD between all possible symmetrical atom matches
    coordinates_lig1 = coordinates_lig1[mas1]
    list_rmsd = []
    for match1 in mas2_list:
        coordinates_lig2_tmp = coordinates_lig2[list(match1)]
        diff = coordinates_lig2_tmp - coordinates_lig1
        list_rmsd.append(np.sqrt((diff * diff).sum() / len(coordinates_lig2_tmp)))  # rmsd
    # Return the minimum RMSD
    lig_rmsd = min(list_rmsd)
    # Write out a PDB file with matched atom names
    if rename_lig2:
        mas2 = mas2_list[np.argmin(list_rmsd)]
        correspondence_key2_item1 = dict(zip(mas2, mas1))
        atom_names_lig1 = [atom1.GetPDBResidueInfo().GetName() for atom1 in lig1.GetAtoms()]
        lig1_ResName = lig1.GetAtoms()[0].GetPDBResidueInfo().GetResidueName()
        for i, atom1 in enumerate(lig2.GetAtoms()):
            atom1.GetPDBResidueInfo().SetResidueName(lig1_ResName)
            if i in correspondence_key2_item1.keys():
                atom1.GetPDBResidueInfo().SetName(atom_names_lig1[correspondence_key2_item1[i]])
        Chem.MolToPDBFile(lig2, output_filename)
    return lig_rmsd

def symm_rmsd(
        nxg: nx.Graph,
        ha_mask: np.ndarray, # = (atomtype != 12) or (atomtype != 1)
        target_pos: np.ndarray,
        pred_pos: np.ndarray,
        time_limit: int = 600,
) -> np.ndarray:
    """
    Args:
        target_pos: shape (N, 3)
        pred_pos: shape (N_pose, N_traj, N, 3)
    """
    try:
        with tm_fn(time_limit):
            swicth_indices = match_graphs(
                nxg, nxg,
                keep_self = True,
            )
    except Exception as e:
        message = f'Find error message during graph matching: {e}.'
        print(message, '\nTurn to use self-graph...')
        num_nodes = nxg.number_of_nodes()
        swicth_indices = [(np.arange(num_nodes), np.arange(num_nodes))]

    # tar already permute
    rmsd_list = []
    for ind, tar in swicth_indices:
        ind_mask = ha_mask[ind]
        comask = ind_mask & ha_mask
        ind = ind[comask]
        tar = tar[comask]
        _target_pos = target_pos[tar]
        _target_pos = torch.from_numpy(_target_pos)
        ind = torch.from_numpy(ind)
        ind = ind.repeat(*(pred_pos.shape[:-2] + (1,) * len(ind.shape)))
        _pred_pos = torch.from_numpy(pred_pos)
        _pred_pos = batched_gather(
            _pred_pos,
            ind,
            dim = -2,
            batch_ndims = len(_pred_pos.shape[:-2])
        )
        diff = _pred_pos - _target_pos.view(*((1, ) * len(_pred_pos.shape[:-2]) + _target_pos.shape))
        rmsd = torch.sqrt(diff.square().sum(dim = -1).mean(dim = -1))
        rmsd_list.append(rmsd)
    rmsd = torch.stack(rmsd_list, dim = 0)
    rmsd = torch.amin(rmsd, dim = 0)

    return rmsd.numpy()

def calc_rmsd(
        mol1: Chem.rdchem.Mol,
        mol2: Chem.rdchem.Mol,
) -> float:
    mol1 = Chem.RemoveAllHs(mol1, sanitize = False)
    mol2 = Chem.RemoveAllHs(mol2, sanitize = False)
    try:
        rmsd_value = get_symmetry_rmsd(
            mol1,
            mol1.GetConformer(0).GetPositions(),
            mol2.GetConformer(0).GetPositions(),
            mol2,
        )
    except Exception as e:
            rmsd_value = CalcLigRMSD(mol1, mol2, removeHs = False)

    return rmsd_value

if __name__ == "__main__":
    import copy
    from druglib.utils.bio_utils import simple_conformer_generation

    # smiles = "CCSc1nnc(NC(=O)CSc2ncccn2)s1"
    # smiles = 'C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1C[C@@H](CCCI)[C@@H]2O'
    smiles = 'COc1cc2[nH]c(Cc3ccccc3)c(C=O)c2cc1-c1cnco1'
    # smiles = 'Cn1nnc(-c2ccc3c(c2)c(C2CCN(CCN4CCOC4=O)CC2)cn3-c2ccc(F)cc2)n1'
    # smiles = 'CCOC(=O)c1cc2c(=O)n3ccccc3nc2n(CCOC)/c1=N\C(=O)C(C)(C)C'
    mol = Chem.MolFromSmiles(smiles)
    mol1 = simple_conformer_generation(copy.deepcopy(mol))
    mol1 = Chem.RemoveHs(mol1)
    mol2 = simple_conformer_generation(copy.deepcopy(mol))
    mol2 = Chem.RemoveHs(mol2)
    pos1 = mol1.GetConformer(0).GetPositions()
    pos2 = mol2.GetConformer(0).GetPositions()
    print('nx graph matching sym RMSD: ', calc_rmsd_nx(mol1, mol2))
    print('pyrmsd sym RMSD: ', get_symmetry_rmsd(mol1, pos1, pos2, mol2))
    print('calc_rmsd sym RMSD: ', calc_rmsd(mol1, mol2))
    print('RDKit sym RMSD: ', CalcLigRMSD(mol1, mol2))