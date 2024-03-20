# Copyright (c) MDLDrugLib. All rights reserved.
import copy
from typing import List, Tuple, Sequence, Optional
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdmolops, rdMolTransforms, AllChem
from scipy.optimize import differential_evolution
from scipy.stats import truncnorm


strict_torsion_smarts = "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])("\
                        "[CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]="\
                        "[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#"\
                        "*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])"\
                        "[CH3])]"
nonstrict_torsion_smarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"

anglesmarts = '*~*~*'


def get_rotatable_bonds(
        mol: Chem.rdchem.Mol,
        strict: bool = True,
) -> List[Tuple[int, int]]:
  """
  Find which bonds are rotatable. Rotatable bonds in
    [(from_atom_id, to_atom_id), ...]
  Args:
      mol: :obj:`Chem.rdchem.Mol`.
      strict: bool, optional. See details in
      https://github.com/rdkit/rdkit/blob/f4529c910e546af590c56eba01f96e9015c269a6/Code/GraphMol/Descriptors/Lipinski.cpp#L107
      It effects the torsion smarts. Default to True.
  Returns:
      rotatable_bonds: List[List[int, int]] List of rotatable bonds in molecule
  """
  if strict:
      torsion_smarts = strict_torsion_smarts
  else:
      torsion_smarts = nonstrict_torsion_smarts
  pattern = Chem.MolFromSmarts(torsion_smarts)
  rdmolops.FastFindRings(mol)
  rotatable_bonds = mol.GetSubstructMatches(pattern)

  return rotatable_bonds

def get_angles(
        mol: Chem.rdchem.Mol
) -> List[Tuple[int, ...]]:
    """
    Find angles consists of three atoms, in [(atom_i id, atom_j id, atom_k id), ...]
    Args:
        mol: :obj:`Chem.rdchem.Mol`.
    """
    smarts = anglesmarts
    pattern = Chem.MolFromSmarts(smarts)
    return mol.GetSubstructMatches(pattern)

def set_dihedral(
        conf,
        idx: Sequence,
        value: float,
) -> None:
    """
    Set single rdkit 3D geometry conformation dihedral angle
    Args:
        conf: rdkit conformer with 3D position.
        idx: sequence. The atoms idx of the dihedral from atom i, j, k, l.
        value: float. The updated number of dihedral.
    """
    assert len(idx) == 4, "Dihedral angles consist of four atoms, i, j, k, l, " \
                          f"but got length of {len(idx)}"
    rdMolTransforms.SetDihedralRad(conf, idx[0], idx[1], idx[2], idx[3], value)

def get_dihedral(
        conf,
        idx: Sequence,
) -> float:
    """
    Get specified dihedral angle from 3D geometry conformation of rdkit molecule.
    Args:
        conf: rdkit conformer with 3D position.
        idx: Sequence. The atoms idx of the dihedral from atom i, j, k, l.
    """
    assert len(idx) == 4, "Dihedral angles consist of four atoms, i, j, k, l, " \
                          f"but got number of {len(idx)}"
    return rdMolTransforms.GetDihedralRad(conf, idx[0], idx[1], idx[2], idx[3])

def set_angle(
        conf,
        idx: Sequence,
        value: float,
) -> None:
    """
    Set single rdkit 3D geometry conformation bend angle
    Args:
        conf: rdkit conformer with 3D position.
        idx: Sequence. The atoms idx of the bend angle from atom i, j, k.
        value: float. The updated number of bend angle.
    """
    assert len(idx) == 3, "Bend angles consist of three atoms, i, j, k, " \
                          f"but got length of {len(idx)}"
    rdMolTransforms.SetAngleRad(conf, idx[0], idx[1], idx[2], value)

def get_angle(
        conf,
        idx: Sequence,
) -> float:
    """
    Get specified bend angle from 3D geometry conformation of rdkit molecule.
    Args:
        conf: rdkit conformer with 3D position.
        idx: Sequence. The atoms idx of the bend angle from atom i, j, k.
    """
    assert len(idx) == 3, "Bend angles consist of three atoms, i, j, k, " \
                          f"but got number of {len(idx)}"
    return rdMolTransforms.GetAngleRad(conf, idx[0], idx[1], idx[2])

def set_bond_length(
        conf,
        idx: Sequence,
        value: float,
) -> None:
    """
    Set single rdkit 3D geometry conformation bond length
    Args:
        conf: rdkit conformer with 3D position.
        idx: Sequence. The atoms idx of the bond from atom i, j.
        value: float. The updated value of bond length.
    """
    assert len(idx) == 2, "Bond consist of two atoms, i, j, " \
                          f"but got number of {len(idx)}"
    rdMolTransforms.SetBondLength(conf, idx[0], idx[1], value)

def get_bond_length(
        conf,
        idx: Sequence,
) -> float:
    """
    Get single rdkit 3D geometry conformation bond length
    Args:
        conf: rdkit conformer with 3D position.
        idx: Sequence. The atoms idx of the bond from atom i, j.
    """
    assert len(idx) == 2, "Bond consist of two atoms, i, j, " \
                          f"but got number of {len(idx)}"
    return rdMolTransforms.GetBondLength(conf, idx[0], idx[1])

def get_mol_dihedrals(
        mol: Chem.rdchem.Mol,
        strict: bool = True,
        remove_hydrogens: bool = False,
        get_all: bool = True,
        bidirectional: bool = False,
) -> Tuple[List[tuple], List[float]]:
    """
    Obtain a single rdkit molecule's dihedral torsions.
    Actually, set molecule dihedrals is more complex.
    And the idx order is important for dihedrals setting,
    When dihedral like as atom_i id, atom_j id, atom_k id, atom_l id,
        it means (atom_j id, atom_k id, atom_l id) bend angle will be
        changed with the updated value while fixed  (atom_i id, atom_j id, atom_k id).
    Args:
        mol: Chem.rdchem.Mol.
        strict: bool, optional. It effects the torsion smarts.
            Default to True.
        remove_hydrogens: bool, optional. Whether dihedrals consisting
            of removing hydrogen.
            Default to False
        get_all: bool, optional. For a rotatable bond, there are more than one
            dihedrals. When we build rdkit molecule using dihedrals, actually one
            dihedral works well while more dihedrals will lead to confusion when the
            model is not work very well.
            So return single one dihedral per rotatable bond if `get_all` set to Fasle,
                otherwise, all per rotatable bond will be returned.
            And when 'get_all' set to False, we notice that we should fix ring side, so the
                other side will be changed while the ring side can be fixed.
            So, `get_all` set to True seems to get regular molecule dihedrals property and
                 `get_all` set to False is suggested when you build a rdkit molecule.
        bidirectional: bool, optional. edge property. It works when `get_all` set to True.
    Returns:
        dihedral ids: a list of tuple (atom_i id, atom_j id, atom_k id, atom_l id)
            with the length of torsion angle number.
        dihedral values: a list of dihedral angle values with the same length as the above
            `dihedral ids`.
    """
    conf = mol.GetConformer()
    torsion_bonds = get_rotatable_bonds(mol, strict = strict)
    dihedrals = []
    dihedral_values = []
    for tbond in torsion_bonds:
        atomj_id, atomk_id = tbond
        bond = mol.GetBondBetweenAtoms(atomj_id, atomk_id)
        atomj = mol.GetAtomWithIdx(atomj_id)
        atomk = mol.GetAtomWithIdx(atomk_id)
        for b_j in atomj.GetBonds():
            # If the same bond as already found torsion angle, then pass it
            if b_j.GetIdx() == bond.GetIdx():
                continue
            # get atom i idx
            atomi_id = b_j.GetOtherAtomIdx(atomj_id)
            for b_k in atomk.GetBonds():
                if (b_k.GetIdx() == bond.GetIdx()
                    or b_k.GetIdx() == b_j.GetIdx()):
                    continue
                # get atom l idx
                atoml_id = b_k.GetOtherAtomIdx(atomk_id)
                # skip 3-membered rings
                if atomi_id == atoml_id:
                    continue
                if (mol.GetAtomWithIdx(atomi_id).GetAtomicNum() == 1
                or mol.GetAtomWithIdx(atoml_id).GetAtomicNum() == 1) and \
                    remove_hydrogens:
                    continue
                if get_all:
                    dihedral_idx = (atomi_id, atomj_id, atomk_id, atoml_id)
                    dihedrals.append(dihedral_idx)
                    dihedral_values.append(get_dihedral(conf, dihedral_idx))
                    if bidirectional:
                        dihedral_idx = (atoml_id, atomk_id, atomj_id, atomi_id)
                        dihedrals.append(dihedral_idx)
                        dihedral_values.append(get_dihedral(conf, dihedral_idx))
                else:
                    # fixed Ring side and rotate the bend angle of the other side
                    # so change the dihedral to target value.
                    if mol.GetAtomWithIdx(atoml_id).IsInRing():
                        dihedral_idx = (atoml_id, atomk_id, atomj_id, atomi_id)
                        dihedrals.append(dihedral_idx)
                        dihedral_values.append(get_dihedral(conf, dihedral_idx))
                        break
                    else:
                        dihedral_idx = (atomi_id, atomj_id, atomk_id, atoml_id)
                        dihedrals.append(dihedral_idx)
                        dihedral_values.append(get_dihedral(conf, dihedral_idx))
                        break
            if not get_all:
                # only get one dihedral per rotatable bond
                break
    return dihedrals, dihedral_values

def get_multi_mols_dihedrals(
        mols: List[Chem.rdchem.Mol],
        strict: bool = True,
        remove_hydrogens: bool = False,
        get_all: bool = True,
        bidirectional: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Apply :func:`get_mol_dihedrals` to multiple molecules
    Args:
        mols: a list of :obj:`rdkit.Chem.rdchem.Mol`.
    Returns:
        dihedral idx array: shape (N, 4), where N is
            cumulative sums of atom numbers.
        dihedral value array: shape (N, ).
        ptr: shape (n + 1, ), where n is the number of input
            args `mols`.
    """
    dihedrals_list = []
    dihedral_values_list = []
    molNum_list = []
    for mol in mols:
        dihedrals, dihedral_values = get_mol_dihedrals(
            mol,
            strict = strict,
            remove_hydrogens = remove_hydrogens,
            get_all = get_all,
            bidirectional = bidirectional
        )
        dihedrals_list.append(dihedrals)
        dihedral_values_list.append(dihedral_values)
        molNum_list.append(mol.GetNumAtoms())

    ptr = _cumsum(molNum_list)
    values = [
        np.array(dihedral_ids, dtype = np.int32) + mol_num
        for dihedral_ids, mol_num in zip(dihedrals_list, ptr[:-1])
    ]
    dihedral_values_arr = np.concatenate(
        [np.array(v, dtype = np.float32) for v in dihedral_values_list],
        axis = 0,
        dtype = np.float32,
    )
    return np.concatenate(values, axis = 0, dtype = np.int32), dihedral_values_arr, ptr


def get_torsion_angles(
        mol: Chem.rdchem.Mol,
) -> List[Tuple[int, ...]]:
    """
    Get the representative torsion angles consisting of
        four atoms. We just consider the torsion angle
        connecting the two isolated fragment. For torsion angle
        in the ring, we ignore it because the ring conformer is mostly fixed.
    We can sample the tautomers and stereoisomers by extra software such as
        schrodinger's LigPrep module, so we can totally focus on the torsion flexibilities
    Args:
        mol: RDKit mol obj.
    Returns:
        Representative torsion angles consisting of four atoms idx. List of four ele tuple.
    """
    G = nx.Graph()
    for i, at in enumerate(mol.GetAtoms()):
        G.add_node(i)
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
    torsions_list = []
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2):
            continue
        l = list(sorted(nx.connected_components(G2), key = len)[0])
        if len(l) < 2:
            continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append(
            (n0[0], e[0], e[1], n1[0])
        )
    return torsions_list

def match_mol_to_ref(
        prb_mol: Chem.rdchem.Mol,
        ref_mol: Chem.rdchem.Mol,
        rotable_bonds: List[Tuple[int, ...]],
        probe_id: int = -1,
        ref_id: int = -1,
        seed: int = 0,
        popsize: int = 15,
        maxiter: int = 500,
        mutation: Tuple[float] = (0.5, 1),
        recombination: float = 0.8,
        distrib: Optional[str] = None,
        **kwargs,
) -> Chem.rdchem.Mol:
    """
    Reference from https://github.com/gcorso/DiffDock/blob/main/datasets/conformer_matching.py#L30
    Args:
        prb_mol: RDkit FF-optimized mol obj to be matched.
        ref_mol: reference mol obj.
        rotable_bonds: torsion angle consisting of four atoms idx
            from the :func:`get_torsion_angles`.
        probe_id: int. ID of the conformation in the probe to be used
            for the alignment. Defaults to first conformation.
        ref_id: int. ID of the conformation in the ref molecule to which
            the alignment is computed. Defaults to first conformation.
    Returns:
        torsion changed probe mol obj, with the optimal aligned RMSD
    Note that the prb_mol does not have to be same with ref_mol.
    An attempt is made to generate one by substructure matching if not the same.
    The output mol haven' t been align to ref_mol in the 3D conformer, the optimization
        process focus on the aligned RMSD to change the torsion of prb_mol.
    """
    object_fn = OptimizeConformer(
        prb_mol, ref_mol, rotable_bonds,
        probe_id, ref_id, seed, distrib, **kwargs)
    bounds = [(-np.pi, np.pi) for _ in range(len(rotable_bonds))]
    minimum = differential_evolution(
        object_fn.score_conformation, bounds,
        maxiter = maxiter, popsize = popsize,
        mutation = mutation, recombination = recombination,
        seed = seed)
    prb_mol = apply_torsion_changes(prb_mol, minimum['x'], rotable_bonds, probe_id)

    return prb_mol

def apply_torsion_changes(
        mol: Chem.rdchem.Mol,
        values: List[float],
        rotable_bonds: List[Tuple[int, ...]],
        conf_id: int = -1,
) -> Chem.rdchem.Mol:
    for i, r in enumerate(rotable_bonds):
        set_dihedral(mol.GetConformer(conf_id), r, values[i])
    return mol

class OptimizeConformer:
    def __init__(
            self,
            prb_mol: Chem.rdchem.Mol,
            ref_mol: Chem.rdchem.Mol,
            rotable_bonds: List[Tuple[int, ...]],
            probe_id: int = -1,
            ref_id: int = -1,
            seed: Optional[int] = None,
            distrib: Optional[str] = None,
            **kwargs,
    ):
        super(OptimizeConformer, self).__init__()
        if seed:
            np.random.seed(seed)
        self.rotable_bonds = rotable_bonds
        self.prb_mol= prb_mol
        self.ref_mol = ref_mol
        self.probe_id = probe_id
        self.ref_id = ref_id

        if distrib is None:
            distrib = 'dirac'
        assert distrib in ['dirac', 'truncnorm', 'constant', 'uniform']
        self.distrib = distrib
        self._define_rmsd_target(**kwargs)

    def _define_rmsd_target(self, **kwargs):
        if self.distrib == 'dirac':
            self.mu = 0
        elif self.distrib == 'truncnorm':
            scale = kwargs['scale']
            sigma = 2 / scale
            dist = truncnorm(-scale, scale, loc = 0, scale = sigma)
            self.mu = abs(dist.rvs(1)[0])
        elif self.distrib == 'constant':
            self.mu = kwargs['constant']
        elif self.distrib == 'uniform':
            self.mu = np.random.uniform(0, 2, 1)[0]
        else:
            raise NotImplementedError(f'Find unsupported rmsd target Sampler called `{self.distrib}`')

    def score_conformation(self, values) -> float:
        self.prb_mol = apply_torsion_changes(
            self.prb_mol, values, self.rotable_bonds, self.probe_id)
        RMSD = AllChem.AlignMol(self.prb_mol, self.ref_mol, self.probe_id, self.ref_id)

        return abs(RMSD - self.mu)

def align_mol(
        prb_mol: Chem.rdchem.Mol,
        ref_mol: Chem.rdchem.Mol,
):
    """
    This function requires the input prb_mol and ref_mol is same in molecule topology
        or say the same mol with different conformer.
    """
    assert ref_mol.GetNumConformers() == 1, 'The ref_mol is required to have only one conformer.'
    ref_mol = copy.deepcopy(ref_mol)
    ref_mol.AddConformer(prb_mol.GetConformer())
    RMSlist = []
    AllChem.AlignMolConformers(ref_mol, RMSlist = RMSlist)
    prb_mol.RemoveAllConformers()
    prb_mol.AddConformer(ref_mol.GetConformers()[1])

    return prb_mol, RMSlist[0]

def get_mol_angles(
        mol: Chem.rdchem.Mol,
        bidirectional: bool = False,
) -> Tuple[List[tuple], List[float]]:
    """
    Obtain a single rdkit molecule's angles.
    Args:
        mol: Chem.rdchem.Mol.
        bidirectional: bool, optional. edge property.
    Returns:
        angle ids: a list of tuple (atom_i id, atom_j id, atom_k id)
            with the length of bend angle number.
        angle values: a list of bend angle values with the same length as the above
            `angle ids`.
    """
    conf = mol.GetConformer()
    Angles = get_angles(mol)
    angles = []
    values = []
    for angle in Angles:
        atomi_id, atomj_id, atomk_id = angle
        angles.append(angle)
        values.append(get_angle(conf, angle))
        if bidirectional:
            angles.append((atomk_id, atomj_id, atomi_id))
            values.append(get_angle(conf, (atomk_id, atomj_id, atomi_id)))

    return angles, values

def get_multi_mols_angles(
        mols: List[Chem.rdchem.Mol],
        bidirectional: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Apply :func:`get_mol_angles` to multiple molecules
    Args:
        mols: a list of :obj:`rdkit.Chem.rdchem.Mol`.
        ignore_Hs: Whether or not ignore the hydrogen atoms.
            Defaults to True.
    Returns:
        angle idx array: shape (N, 3), where N is
            cumulative sums of atom numbers.
        angle value array: shape (N, ).
        ptr: shape (n + 1, ), where n is the number of input
            args `mols`.
    """
    angles_list = []
    angle_values_list = []
    molNum_list = []
    for mol in mols:
        angles, angle_values = get_mol_angles(
            mol,
            bidirectional = bidirectional
        )
        angles_list.append(angles)
        angle_values_list.append(angle_values)
        molNum_list.append(mol.GetNumAtoms())
    ptr = _cumsum(molNum_list)
    values = [
        np.array(angle_ids, dtype = np.int32) + mol_num
        for angle_ids, mol_num in zip(angles_list, ptr[:-1])
    ]
    angle_values_arr = np.concatenate(
        [np.array(v, dtype = np.float32) for v in angle_values_list],
        axis = 0,
        dtype = np.float32,
    )
    return np.concatenate(values, axis = 0, dtype = np.int32), angle_values_arr, ptr

def get_mol_bonds(
        mol: Chem.rdchem.Mol,
        bidirectional: bool = False,
) -> Tuple[List[tuple], List[float]]:
    """
    Obtain a single rdkit molecule's bonds.
    Args:
        mol: Chem.rdchem.Mol.
        bidirectional: bool, optional. edge property.
    Returns:
        bond ids: a list of tuple (atom_i id, atom_j id)
            with the length of bond number.
        bond length values: a list of bond length values with
            the same length as the above `bond ids`.
    """
    conf = mol.GetConformer()
    bonds = []
    dist = []
    for b in mol.GetBonds():
        idx = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        bonds.append(idx)
        dist.append(get_bond_length(conf, idx))
        if bidirectional:
            bonds.append(idx[::-1])
            dist.append(get_bond_length(conf, idx[::-1]))

    return bonds, dist

def get_multi_mols_bonds(
        mols: List[Chem.rdchem.Mol],
        bidirectional: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Apply :func:`get_mol_bonds` to multiple molecules
    Args:
        mols: a list of :obj:`rdkit.Chem.rdchem.Mol`.
    Returns:
        bond idx array: shape (N, 2), where N is
            cumulative sums of atom numbers.
        bond langth array: shape (N, ).
        ptr: shape (n + 1, ), where n is the number of input
            args `mols`.
    """
    bonds_list = []
    bond_length_list = []
    molNum_list = []
    for mol in mols:
        bonds, bond_lengths = get_mol_bonds(
            mol,
            bidirectional = bidirectional
        )
        bonds_list.append(bonds)
        bond_length_list.append(bond_lengths)
        molNum_list.append(mol.GetNumAtoms())
    ptr = _cumsum(molNum_list)
    values = [
        np.array(bond_ids, dtype = np.int32) + mol_num
        for bond_ids, mol_num in zip(bonds_list, ptr[:-1])
    ]
    bond_lengths_arr = np.concatenate(
        [np.array(v, dtype = np.float32) for v in bond_length_list],
        axis = 0,
        dtype = np.float32,
    )
    return np.concatenate(values, axis = 0, dtype = np.int32), bond_lengths_arr, ptr

def mol_with_atom_index(
        mol: Chem.rdchem.Mol,
) -> Chem.rdchem.Mol:
    """
    For every atom in the input molecule, set property `molAtomMapNumber`.
    Args & Returns:
        mol: Chem.rdchem.Mol.
    """
    for idx in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(idx).SetProp(
            'molAtomMapNumber',
            str(mol.GetAtomWithIdx(idx).GetIdx())
        )
    return mol

def atom_env(
        mol: Chem.rdchem.Mol,
        maxradius: int = 3,
) -> List[str]:
    """
    Get atom environments to smarts.
    Args:
        mol: Chem.rdchem.Mol.
        maxradius: int, topological path, represents the edge number.
            Default to 3.
    Returns:
        a list of smarts of every atom's environment (or fragment including the main atom).
    """
    envs = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        env = Chem.FindAtomEnvironmentOfRadius(mol, maxradius, idx)
        atommap = {}
        submol = Chem.PathToSubmol(mol, env, atomMap = atommap)
        if atommap.get(idx) is not None:
            envs.append(Chem.MolToSmarts(submol))
    return envs




########################## helper function

def _cumsum_wo_last(
        data_list: List[int],
        offset: int = 0,
) -> np.ndarray:
    """
    Returns cumulative sums for a set of counts, removing last item.
    This helpher function is useful for reindexing.
    Args:
        data_list: list. Typically, data_list is a list of int data.
        offset: int, optional. Extra offset for every cumulative sum.
    Returns:
        a list with the same length as input data list.
    E.g.:
        >>> a = [1, 2, 3]
        >>> _cumsum_wo_last(a)
        [0, 1, 3]# without the last one, 5
    """
    return np.delete(np.insert(np.cumsum(data_list, dtype = np.int32), 0, 0), -1) + offset

def _cumsum(
        data_list: List[int],
        offset: int = 0,
) -> np.ndarray:
    """
    Returns cumulative sums for a set of counts, keeping last item.
    This helpher function is useful for reindexing.
    Args:
        data_list: list. Typically, data_list is a list of int data.
        offset: int, optional. Extra offset for every cumulative sum.
    Returns:
        a list with the length as input data list plus one.
    E.g.:
        >>> a = [1, 2, 3]
        >>> _cumsum(a)
        [0, 1, 3, 5]
    """
    return np.insert(np.cumsum(data_list, dtype = np.int32), 0, 0) + offset


