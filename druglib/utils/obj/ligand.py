# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Optional,
    Union,
    List,
    Tuple,
    Mapping,
)
import copy
import os.path as osp
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor

from rdkit import Chem, Geometry, rdBase, RDLogger
from rdkit.Geometry import Point3D
from rdkit.Chem import GetPeriodicTable
pt = GetPeriodicTable()

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')

from druglib.alerts.molerror import MolReconstructError
from . import ligand_constants as lc
from .ligand_math import merge_edge
from ..torch_utils import get_complete_subgraph
# from druglib.utils.bio_utils import (
#     read_mol, get_atom_partial_charge, simple_conformer_generation,
#     get_mol_angles, remove_all_hs, get_ligconf
# )
from druglib.utils import bio_utils


@dataclass(frozen = True)
class Ligand3D:
    """Ligand Topology and Geometry representation"""
    # the :obj:Ligand3D source name (file name or compound name)
    name: str
    # the number of atoms
    numatoms: int
    # the number of bonds, plus connectivity number
    # the same as the num_edges
    numbonds: int

    # atom type idx: node matrix, with explicit hydrogen atoms is better
    atomtype: np.ndarray  # [num_atom]

    # edge index
    edge_index: np.ndarray # [2, num_edge] undirected graph

    # bond type: edge attr
    bond_type: np.ndarray # [num_edge]

    # Cartesian coordinates of atoms in angstroms. Note that the
    # given molecule does not necessarily have a 3D geometry,
    # we try to use the rdkit force field to generate the 3D conformer
    # [num_atoms, 3] or None
    atom_positions: Optional[np.ndarray] = None

    # the center of mass
    cm: np.ndarray = None # [3]

    ########################### save property value, if able to one-hot encoding,
    ########################### using the one-hot encoding.
    # atom property: every atom property
    atom_prop: Optional[Mapping] = None

    # bond property: every bond property (chemical bond or connectivity)
    bond_prop: Optional[Mapping] = None

    # rdkit.Chem.rdchem.Mol object
    model: Optional[Chem.rdchem.Mol] = None

    def _reconstruct(self) -> Chem.rdchem.Mol:
        return reconstruct(
            self.atomtype,
            self.atom_positions,
            self.edge_index,
            self.bond_type,
            self.atom_prop.get('formal_charge', None),
            True,
        )

    def to_sdf(self, output: str):
        """Export the :obj:`Ligand3D` to .SDF file"""
        sdf_writer = Chem.SDWriter(output)

        mol = self.model
        if mol is None:
            mol = self._reconstruct()

        if mol.GetNumConformers() > 0:
            for conf in mol.GetConformers():
                # save the 3D conformer of molecule
                sdf_writer.write(mol, confId = conf.GetId())
        else:
            # save the topology of molecule.
            sdf_writer.write(mol)

    def pos_update(
            self,
            new_pos: Union[np.ndarray, torch.Tensor],
            model: Optional[Chem.rdchem.Mol] = None,
            hasRemovedHs: bool = False,
    ) -> 'Ligand3D':
        """
        Supported case:
            1). Ligand heavy atoms coordinates new position update from original mol object
                with explicit Hs (heavy atoms array position equal to rdkit atoms iteration sequence);
            2) RemoveHs rdkit position array
        """
        if isinstance(new_pos, torch.Tensor):
            new_pos = new_pos.cpu().numpy()
        new_shape = new_pos.shape
        assert len(new_shape) == 2, f'New position shape must be (N_atoms, 3), but got shape {new_shape}'

        if model is None:
            model = copy.deepcopy(self.model)
            numconf = model.GetNumConformers()
            if numconf == 0:
                if hasRemovedHs:
                    model = Chem.AddHs(model)
                model = bio_utils.simple_conformer_generation(model)
                if hasRemovedHs:
                    model = Chem.RemoveHs(model)
            conf = model.GetConformer()
            count = 0
            for at_id, at in enumerate(model.GetAtoms()):
                if not hasRemovedHs and at.GetSymbol() == 'H':
                    continue
                x, y, z = new_pos[count]
                conf.SetAtomPosition(at_id, Point3D(float(x), float(y), float(z)))
                count += 1
            if not hasRemovedHs:
                model = bio_utils.remove_all_hs(model)
                model = Chem.AddHs(model, addCoords = True)

        if 'atweight' in self.atom_prop:
            cm = (new_pos * self.atom_prop['atweight'].reshape(-1, 1)).sum(axis = 0) / self.atom_prop['atweight'].sum()
        else:
            cm = np.mean(new_pos, axis = 0)

        return Ligand3D(
            name = self.name,
            numatoms = self.numatoms,
            numbonds = self.numbonds,
            atomtype = self.atomtype,
            edge_index = self.edge_index,
            bond_type = self.bond_type,
            atom_positions = new_pos,
            cm = cm,
            atom_prop = self.atom_prop,
            bond_prop = self.bond_prop,
            model = model,
        )

    def _merge_edge(
            self,
            extra_edge: np.ndarray,
    ):
        """
        Helper function: Add extra edge into the `edge_index` with an another class.
        Merge the original chemical bond `self.edge_index` with the input `extra_edge`
            also called geometry graph connectivity.
        Args:
            extra_edge: np.ndarray. Shape (2, num_edges)
            Note that must be from the same :obj:Chem.rdchem.Mol
            and the atom idx must be the same between the edge_index
            and `extra_edge`.
        Returns:
            new edge_index and edge_attr (using connect_types idx)
            new_add_mask: Indicate (1) which edges are the new compared with
                original `edge_index`.
        E.g.:
            >>> edge_index = np.array([[1, 1, 2, 3],
                                       [2, 3, 4, 2]])# self.edge_index
            >>> N = 5# self.numatoms
            >>> bond_type = np.array([1, 0, 2, 0])# self.bond_type
            >>> extra_edge = np.array([[0, 4, 1, 1],
                                       [1, 0, 2, 3]])
            >>> # merge_edge(..., extra_edge)# lc.connect_to_id['NoneType'] -> 10
            (array([[0, 1, 1, 2, 3, 4],
                    [1, 2, 3, 4, 2, 0]]),
             array([10,  1,  0,  2,  0, 10]),
             array([1,  0,  0,  0,  0, 1]))
        """
        new_edge_index, new_bond_type, mask = merge_edge(
            extra_edge = extra_edge,
            edge_index = self.edge_index,
            bond_type = self.bond_type,
            num_nodes = self.numatoms,
        )
        return new_edge_index, new_bond_type, mask

    def get_ring_graph(
            self,
            allow_merge: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the ring info of :attr:`model` or
            reconstructed :obj:`Chem.rdchem.Mol`.
        Args:
            allow_merge: bool, optional. Whether or not merge the ring graph
                with the original chemical bond `self.edge_index`.
            Defaults to True, allow to merge.
        Returns:
            new edge_index with extra 'NoneType' class, representing the long-range
                interations and constraints from 3D geometric view.
            edge_attr, also called bond type. Shape (num_edges, ) using connect_types idx
            new_add_mask: Indicate (1) which edges are the new compared with
                original `edge_index`.
        """
        mol = self.model
        if mol is None:
            # try to reconstruct the :obj:Chem.rdchem.Mol
            mol = self._reconstruct()

        ssr = Chem.GetSymmSSSR(mol)

        if len(ssr) < 0.5:
            if allow_merge:
                return self.edge_index, self.bond_type, np.zeros_like(self.bond_type)
            else:
                return np.empty(shape = (2, 0)), np.empty(shape = 0), np.empty(shape = 0)
        ring_subgraphs = []
        for ring in ssr:
            ring_subgraphs.append(
                get_complete_subgraph(np.array(ring))
            )
        ring_graph = np.concatenate(ring_subgraphs, axis = 1)

        if allow_merge:
            edge_index, bond_type, new_add_mask = self._merge_edge(ring_graph)
            return edge_index, bond_type, new_add_mask # no duplications
        else:
            ring_graph = np.unique(ring_graph, axis = 1)
            bond_type = np.full(ring_graph.shape[-1], fill_value = lc.connect_to_id['NoneType'])
            return ring_graph, bond_type, np.ones_like(bond_type, dtype = int) # maybe duplicate with the original edge_index

    def get_two_hop_graph(
            self,
            allow_merge: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get two hop subgraph, related with angles, extracted by getting
            the :obj:Chem.rdchem.Mol angles.
        Args:
            allow_merge: bool, optional. Whether or not merge the two
                hop graph with the original chemical bond `self.edge_index`.
            Defaults to True, allow to merge.
        Returns:
            new edge_index with extra 'NoneType' class, representing the
                angles and constrains.
            edge_attr, also called bond type. Shape (num_edges, ) using connect_types idx
            new_add_mask: Indicate (1) which edges are the new compared with
                original `edge_index`.
        """
        mol = self.model
        if mol is None:
            # try to reconstruct the :obj:Chem.rdchem.Mol
            mol = self._reconstruct()
        angle_index, angle_values = bio_utils.get_mol_angles(
            mol, bidirectional = True)
        if len(angle_values) < 0.5:
            if allow_merge:
                return self.edge_index, self.bond_type, np.zeros_like(self.bond_type)
            else:
                return np.empty(shape = (2, 0)), np.empty(shape = 0), np.empty(shape = 0)
        src = []
        dst = []
        for angle_id in angle_index:
            src.append(angle_id[0])
            dst.append(angle_id[-1])
        two_hop_graph = np.array([src, dst], dtype = int)

        if not allow_merge:
            two_hop_graph = np.unique(two_hop_graph, axis=1)
            bond_type = np.full(two_hop_graph.shape[-1], fill_value = lc.connect_to_id['NoneType'])
            return two_hop_graph, bond_type, np.ones_like(bond_type, dtype = int)  # maybe duplicate with the original edge_index
        else:
            edge_index, bond_type, new_add_mask = self._merge_edge(two_hop_graph)
            return edge_index, bond_type, new_add_mask  # no duplications

    def get_knn_graph(
            self,
            k: int,
            allow_merge: bool = True,
            num_workers: int = 16
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get KNN subgraph, related with atom environments.
        Args:
            k: int. The number of neighbors.
            allow_merge: bool, optional. Whether or not merge the two
                hop graph with the original chemical bond `self.edge_index`.
            Defaults to True, allow to merge.
            num_workers: int. Number of workers to use for computation. Has no
                effect in the input lies on the GPU. Defaults to 16.
        Returns:
            new edge_index with extra 'NoneType' class, representing the
                angles and constrains.
            edge_attr, also called bond type. Shape (num_edges, ) using connect_types idx
            new_add_mask: Indicate (1) which edges are the new compared with
                original `edge_index`.
        """
        pos = self.atom_positions
        if pos is None:
            mol = self._reconstruct()
            pos = mol.GetConformer(0).GetPositions()

        from torch_geometric.nn import knn_graph
        pos = torch.tensor(pos, dtype = torch.float32)
        knn_edge = knn_graph(
            x = pos,
            k = k,
            loop = False,
            num_workers = num_workers,
        )
        if knn_edge.shape[-1] == 0:
            if allow_merge:
                return self.edge_index, self.bond_type, np.zeros_like(self.bond_type)
            else:
                return np.empty(shape = (2, 0)), np.empty(shape = 0), np.empty(shape = 0)

        knn_edge = knn_edge.numpy()
        if not allow_merge:
            bond_type = np.full(knn_edge.shape[-1], fill_value = lc.connect_to_id['NoneType'])
            return knn_edge, bond_type, np.ones_like(bond_type, dtype = int)  # maybe duplicate with the original edge_index
        else:
            edge_index, bond_type, new_add_mask = self._merge_edge(knn_edge)
            return edge_index, bond_type, new_add_mask  # no duplications

def reconstruct(
        atomtype: Union[List[int], np.ndarray, Tensor],
        atom_positions: Union[list, np.ndarray, Tensor, None],
        edge_index: Union[list, np.ndarray, Tensor],
        bond_type: Union[list, np.ndarray, Tensor],
        formal_charge: Union[list, np.ndarray, Tensor, None] = None,
        sanitize: bool = True
) -> Chem.rdchem.Mol:
    """
    Reconstruct the :obj:`Ligand3D` to rdkit.Chem.rdchem.Mol.
    Args:
        atomtype: the atom types of molecule. Shape (num_atom, ).
        atom_positions: the atom 3D position. Shape (num_atom, 3)
        edge_index: the chemical bond of molecule. Shape (2, num_bond) undirected edge.
        bond_type: the chemical bond type. Shape (num_bond, )
        formal_charge: the atom formal charge. Defaults to None.
        sanitize: bool, optional. Whether to sanitize the built :obj:Chem.rdchem.Mol.
            Defaults to True.
    Returns:
        :obj:Chem.rdchem.Mol with 3D conformer.
    """
    # shape check
    assert len(atomtype) == len(atom_positions), "atom type and atom position shape mismatch"
    assert len(edge_index) == 2, "edge_index must be the format of [2, num_edge]"
    assert len(edge_index[0]) == len(bond_type), "the row of edge index and bond type shape mismatch"
    if isinstance(atomtype, np.ndarray):
        atomtype = atomtype.tolist()
    elif isinstance(atomtype, Tensor):
        atomtype = atomtype.clone().cpu().tolist()

    if isinstance(atom_positions, np.ndarray):
        atom_positions = atom_positions.tolist()
    elif isinstance(atom_positions, Tensor):
        atom_positions = atom_positions.clone().cpu().tolist()

    if isinstance(edge_index, np.ndarray):
        edge_index = edge_index.tolist()
    elif isinstance(edge_index, Tensor):
        edge_index = edge_index.clone().cpu().tolist()

    if isinstance(bond_type, np.ndarray):
        bond_type = bond_type.tolist()
    elif isinstance(bond_type, Tensor):
        bond_type = bond_type.clone().cpu().tolist()

    if formal_charge is not None:
        if isinstance(formal_charge, np.ndarray):
            formal_charge = formal_charge.tolist()
        elif isinstance(formal_charge, Tensor):
            formal_charge = formal_charge.clone().cpu().tolist()

    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atomtype))
    for idx, at in enumerate(atomtype):
        atom = Chem.Atom(at)
        mol.AddAtom(atom)
        if formal_charge is not None:
            atom.SetFormalCharge(int(formal_charge[idx]))
        if atom_positions is not None:
            xyz = Geometry.Point3D(*atom_positions[idx])
            conf.SetAtomPosition(idx, xyz)
    if atom_positions is not None:
        mol.AddConformer(conf)

    for i, bt in enumerate(bond_type):
        node_i, node_j = edge_index[0][i], edge_index[1][i]
        if node_i > node_j:
            bd_str = lc.bond_types[bt]
            if bd_str == 'other':
                raise MolReconstructError("Unknown bond order, `other` type")
            mol.AddBond(node_i, node_j, Chem.BondType.__dict__[bd_str])
    mol = mol.GetMol()

    # post-processing: sanitize and kekulize
    if lc.bondtypes_to_id['AROMATIC'] in bond_type:
        Chem.Kekulize(mol, clearAromaticFlags = True)
    if sanitize:
        Chem.SanitizeMol(mol, Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE^Chem.SANITIZE_SETAROMATICITY)

    return mol


def ligand_parser(
        ligand_path: Union[str, Chem.rdchem.Mol],
        sanitize: bool = True,
        calc_charges: bool = True,
        remove_hs: bool = False,
        assign_chirality: bool = False,
        allow_genconf: bool = True,
        emb_multiple_3d: Optional[int] = None,
        only_return_mol: bool = False,
        name: Optional[str] = None,
) -> Optional[Union[Ligand3D, Chem.rdchem.Mol]]:
    """
    Parse a molecule file to :obj:`Ligand3D`.
    Args:
      ligand_path: str. A molecule file in the format of .sdf, .mol2, .pdbqt or .pdb or rdkit mol
      sanitize: bool, optional. rdkit sanitize molecule.
            Default to True.
      calc_charges: bool, optional. If True, add Gasteiger charges.
            Default to True.
            Note that when calculating charges, the mol must be sanitized.
      remove_hs: bool, optional. If True, remove the hydrogens.
            Default to False. We suggest keep the hydrogen atoms.
      assign_chirality: bool, optional. If True, inference the chirality.
            Default to False.
      allow_genconf: bool, optional. If detect no conformer, whether or not
        allow to generate a new conformer.
            Default to True.
      emb_multiple_3d: int, optional. If int, generate multiple conformers for mol.
    Returns:
      A new `Ligand3D` parsed from the molecule file.
    """
    if name is None:
        if not isinstance(ligand_path, Chem.rdchem.Mol):
            name = osp.basename(ligand_path)
        else:
            try:
                mol = ligand_path
                name = mol.GetProp('_Name')
            except AttributeError:
                name = 'Druglib-Mol'

    mol = bio_utils.read_mol(
        mol_file = ligand_path,
        sanitize = sanitize,
        calc_charges = calc_charges,
        remove_hs = remove_hs,
        assign_chirality = assign_chirality,
        emb_multiple_3d = emb_multiple_3d,
    )

    if mol is None:
        return None

    if only_return_mol:
        return mol

    # if remove_hs:
    #     atomtypes = lc.atom_types
    # else:
    atomtypes = lc.atom_types_with_H

    # 1. get topology property
    NA = mol.GetNumAtoms()
    NB = mol.GetNumBonds()
    ring_info = mol.GetRingInfo()

    # 2. data collector
    atom_types = []
    atomprop = defaultdict(list)

    src_list = []
    dst_list = []
    bond_type = []
    bondprop = defaultdict(list)

    # 3. atom iteration
    for idx, at in enumerate(mol.GetAtoms()):
        atid = at.GetIdx()
        assert atid == idx, f"Atom id `{atid}` from rdkit.Chem.rdchem.Mol " \
                            f"is mismatched with iteration id `{idx}`."
        # use PeriodicTable so we can reconstruct the mol from array
        at_number = pt.GetAtomicNumber(at.GetSymbol().capitalize())
        atom_types.append(at_number)

        # use the dense feature rather than one-hot encoding if unnecessary
        # atom element feature build. If not frequent atom, regard it as 'other'
        # because it is rare samples.
        atomprop['symbol'].append(lc.types_index(at.GetSymbol(), atomtypes))
        atomprop['atweight'].append(pt.GetAtomicWeight(at_number))
        atomprop['hybridization'].append(lc.types_index(str(at.GetHybridization()), lc.hybridization_types))
        atomprop['degree'].append(at.GetTotalDegree())
        atomprop['implicit_valence'].append(at.GetImplicitValence())
        atomprop['explicit_valence'].append(at.GetExplicitValence())
        atomprop['numring'].append(ring_info.NumAtomRings(atid))
        atomprop['isaromatic'].append(int(at.GetIsAromatic()))
        atomprop['chirality'].append(lc.types_index(str(at.GetChiralTag()), lc.CHI_chiral_types))
        atomprop['radical'].append(lc.types_index(at.GetNumRadicalElectrons(), lc.num_radical_electrons))
        atomprop['numHs'].append(lc.types_index(at.GetTotalNumHs(includeNeighbors=True), lc.total_num_hydrogen))

        atomprop['formal_charge'].append(at.GetFormalCharge())
        if calc_charges:
            atomprop['partialcharge'].append(bio_utils.get_atom_partial_charge(at))

        # use one-hot encoding like array to represent atom in the multiple rings
        # use the 'vec' KEYWORDS to represent the it is a vector, not a value
        # Be careful in the concat the feature.
        atomprop['isinringn_vec'].append(lc.IsInRingofN(ring_info, atid))

    assert NA == len(atom_types)

    # 'xx_vec' key shape (Num_nodes, F) vs. others (Num_nodes,)
    new_atomprop = {k: np.array(v, dtype = np.float32) for k, v in atomprop.items()}

    # chemical pharmacophore feature
    cf_arr = np.zeros(shape = (NA, len(lc.aff_types)), dtype = np.float32)
    for feat in lc.Factory.GetFeaturesForMol(mol):
        cf_arr[feat.GetAtomIds(), lc.aff_to_id[feat.GetFamily()]] = 1
    new_atomprop['chemfeature_vec'] = cf_arr

    # chiral feature
    if assign_chirality:
        chiral_arr = np.zeros(shape=(NA, len(lc.chiral_types)), dtype = np.float32)
        centers = Chem.FindMolChiralCenters(
            mol,
            force = True,
            useLegacyImplementation = False,
        )
        for (idx, chiral_type) in centers:
            if chiral_type in lc.chiral_types:
                chiral_arr[idx, lc.chiral_types.index(chiral_type)] = 1
            else:
                chiral_arr[idx, -1] = 1
        chiral_arr_sum = chiral_arr.sum(axis = -1)
        mask = (chiral_arr_sum < 0.5)
        chiral_arr[mask, -1] = 1
        new_atomprop['chiral_vec'] = chiral_arr

    # 4. bond iteration
    for idx, bd in enumerate(mol.GetBonds()):
        start, end = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        src_list.extend([start, end])
        dst_list.extend([end, start])
        bond_type.extend(
            [lc.types_index(str(bd.GetBondType()), lc.bond_types)] * 2
        )
        bondprop['bondstereo'].extend(
            [lc.types_index(str(bd.GetStereo()), lc.bondstereo_types)] * 2)
        bondprop['isinring'].extend([int(bd.IsInRing())] * 2)
        bondprop['isconjugated'].extend([int(bd.GetIsConjugated())] * 2)
    assert 'isinring' in bondprop
    edge_index = np.array([src_list, dst_list], dtype = int)
    perm = (edge_index[0] * NA + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    bond_type = np.array(bond_type, dtype = int)[perm]
    new_bondprop = {k: np.array(v, dtype = np.float32)[perm] for k, v in bondprop.items()}
    new_bondprop['bond_label'] = np.zeros(len(bond_type), dtype = int)

    # 5. try to get 3D geometry (/conformer)
    atom_positions = bio_utils.get_ligconf(mol, allow_genconf)

    atom_types = np.array(atom_types, dtype = int)

    cm = (atom_positions * new_atomprop['atweight'].reshape(-1, 1)).sum(axis = 0) / new_atomprop['atweight'].sum()

    return Ligand3D(
        name = name,
        numatoms = NA,
        numbonds = NB,
        atomtype = atom_types,
        edge_index = edge_index,
        bond_type = bond_type,
        atom_positions = atom_positions,
        cm = cm,
        atom_prop = new_atomprop,
        bond_prop = new_bondprop,
        model = mol,
    )
