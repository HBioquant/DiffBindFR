import copy
import warnings
import numpy as np
import networkx as nx

import torch

from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.utils import to_dense_adj, dense_to_sparse
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# orderd by perodic table
atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(7)
num_hs_vocab = range(7)
formal_charge_vocab = range(-5, 6)
chiral_tag_vocab = range(4)
total_valence_vocab = range(8)
num_radical_vocab = range(8)
hybridization_vocab = range(len(Chem.rdchem.HybridizationType.values))

bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1

    return feature

def atom_default(atom):
    """Default atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol dim=18

        GetChiralTag(): one-hot embedding for atomic chiral tag dim=5

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs dim=5

        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetNumRadicalElectrons(): one-hot embedding for the number of radical electrons on the atom

        GetHybridization(): one-hot embedding for the atom's hybridization

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
        18 + 5 + 8 + 12 + 8 + 9 + 10 + 9 + 3 + 4
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetChiralTag(), chiral_tag_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True) + \
           onehot(atom.GetNumRadicalElectrons(), num_radical_vocab, allow_unknown=True) + \
           onehot(atom.GetHybridization(), hybridization_vocab, allow_unknown=True) + \
            onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           [atom.GetIsAromatic(), atom.IsInRing(), atom.IsInRing() and (not atom.IsInRingSize(3)) and (not atom.IsInRingSize(4)) \
            and (not atom.IsInRingSize(5)) and (not atom.IsInRingSize(6))]+[atom.IsInRingSize(i) for i in range(3, 7)]

def bond_default(bond):
    """Default bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond

        GetBondDir(): one-hot embedding for the direction of the bond

        GetStereo(): one-hot embedding for the stereo configuration of the bond

        GetIsConjugated(): whether the bond is considered to be conjugated
    """
    return onehot(bond.GetBondType(), bond_type_vocab, allow_unknown=True) + \
           onehot(bond.GetBondDir(), bond_dir_vocab) + \
           onehot(bond.GetStereo(), bond_stereo_vocab, allow_unknown=True) + \
           [int(bond.GetIsConjugated())]

def get_full_connected_edge(frag):
    frag = np.asarray(list(frag))
    return torch.from_numpy(np.repeat(frag, len(frag)-1)), \
        torch.from_numpy(np.concatenate([np.delete(frag, i) for i in range(frag.shape[0])], axis=0))

def remove_repeat_edges(new_edge_index, refer_edge_index, N_atoms):
    new = to_dense_adj(new_edge_index, max_num_nodes=N_atoms)
    ref = to_dense_adj(refer_edge_index, max_num_nodes=N_atoms)
    delta_ = new - ref
    delta_[delta_<1] = 0
    unique, _ = dense_to_sparse(delta_)
    return unique

def get_ligand_feature(mol, use_chirality=True):
    xyz = mol.GetConformer().GetPositions()
    # covalent
    N_atoms = mol.GetNumAtoms()
    node_feature = []
    edge_index = []
    edge_feature = []
    G = nx.Graph()
    for idx, atom in enumerate(mol.GetAtoms()):
        # node
        node_feature.append(atom_default(atom))
        # edge
        for bond in atom.GetBonds():
            edge_feature.append(bond_default(bond))
            for bond_idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
                if bond_idx != idx:
                    edge_index.append([idx, bond_idx])
                    G.add_edge(idx, bond_idx)
    node_feature = torch.from_numpy(np.asanyarray(node_feature)).float()
    if use_chirality:
        try:
            chiralcenters = Chem.FindMolChiralCenters(
                mol,
                force=True,
                includeUnassigned=True,
                useLegacyImplementation=False
            )
        except:
            chiralcenters = []
        chiral_arr = torch.zeros([N_atoms,3]) 
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] =1 
            elif rs == 'S':
                chiral_arr[i, 1] =1 
            else:
                chiral_arr[i, 2] =1 
        node_feature = torch.cat([node_feature, chiral_arr.float()], dim=1)
    edge_index = torch.from_numpy(np.asanyarray(edge_index).T).long()
    edge_feature = torch.from_numpy(np.asanyarray(edge_feature)).float()
    cov_edge_num = edge_index.size(1)
    l_full_edge_s = edge_feature[:, :5]
    xyz = torch.from_numpy(xyz)
    l_full_edge_s = torch.cat([l_full_edge_s, torch.pairwise_distance(xyz[edge_index[0]], xyz[edge_index[1]]).unsqueeze(dim=-1)], dim=-1)
    # get fragments based on rotation bonds
    frags = []
    rotate_bonds = []
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        l = (sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        else:
            frags.append(l)
    if len(frags) != 0:
        frags = sorted(frags, key=len)
        for i in range(len(frags)):
            for j in range(i+1, len(frags)):
                frags[j] -= frags[i]
        frags = [i for i in frags if len(i) > 1]
        frag_edge_index = torch.cat([torch.stack(get_full_connected_edge(i), dim=0) for i in frags], dim=1).long()
        edge_index_new = remove_repeat_edges(new_edge_index=frag_edge_index, refer_edge_index=edge_index, N_atoms=N_atoms)
        edge_index = torch.cat([edge_index, edge_index_new], dim=1)
        edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
        edge_feature_new[:, [4, 5, 18]] = 1
        edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
        l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], torch.pairwise_distance(xyz[edge_index_new[0]], xyz[edge_index_new[1]]).unsqueeze(dim=-1)], dim=-1)
        l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # interaction
    adj_interaction = torch.ones((N_atoms, N_atoms)) - torch.eye(N_atoms, N_atoms)
    interaction_edge_index, _ = dense_to_sparse(adj_interaction)
    edge_index_new = remove_repeat_edges(new_edge_index=interaction_edge_index, refer_edge_index=edge_index, N_atoms=N_atoms)
    edge_index = torch.cat([edge_index, edge_index_new], dim=1)
    edge_feature_new = torch.zeros((edge_index_new.size(1), 20))
    edge_feature_new[:, [4, 5, 18]] = 1
    edge_feature = torch.cat([edge_feature, edge_feature_new], dim=0)
    interaction_edge_mask = torch.ones((edge_feature.size(0),))
    interaction_edge_mask[-edge_feature_new.size(0):] = 0
    # scale the distance
    l_full_edge_s[:, -1] *= 0.1
    l_full_edge_s_new = torch.cat([edge_feature_new[:, :5], -torch.ones(edge_feature_new.size(0), 1)], dim=-1)
    l_full_edge_s = torch.cat([l_full_edge_s, l_full_edge_s_new], dim=0)
    # cov edge mask
    cov_edge_mask = torch.zeros(edge_feature.size(0),)
    cov_edge_mask[:cov_edge_num] = 1
    x = (xyz, node_feature, edge_index, edge_feature, l_full_edge_s, interaction_edge_mask.bool(), cov_edge_mask.bool()) 
    return x 
