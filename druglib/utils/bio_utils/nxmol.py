# Copyright (c) MDLDrugLib. All rights reserved.
from rdkit import Chem
import networkx as nx


def mol2nx(
        mol: Chem.rdchem.Mol,
) -> nx.Graph:
    """
    rdkit molecule to convert networkx graph.
    Args:
        mol: rdkit mol.
    Returns:
        networkx graph.
    """
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num = atom.GetAtomicNum(),
            formal_charge = atom.GetFormalCharge(),
            chiral_tag = atom.GetChiralTag(),
            hybridization = atom.GetHybridization(),
            num_explicit_hs = atom.GetNumExplicitHs(),
            is_aromatic = atom.GetIsAromatic(),
        )
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type = bond.GetBondType(),
        )

    return G

def nx2mol(
        G: nx.Graph
) -> Chem.rdchem.Mol:
    """
    Molecule formatting as networkx graph to rebuild rdkit mol
    Args:
        G: networkx graph.
    Returns:
        rdkit mol.
    """
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')

    node_to_idx = {}
    for node in G.nodes():
        a = Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        source, target = edge
        idx1 = node_to_idx[source]
        idx2 = node_to_idx[target]
        bond_type = bond_types[source, target]
        mol.AddBond(idx1, idx2, bond_type)

    Chem.SanitizeMol(mol)

    return mol
