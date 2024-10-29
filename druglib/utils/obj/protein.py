# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Optional, Tuple, List, Union,
)
import warnings, string, Bio, copy, pathlib
import os.path as osp
from datetime import date
from dataclasses import dataclass
from easydict import EasyDict as ed
from collections import OrderedDict, defaultdict
import numpy as np
import torch
from torch import Tensor

from Bio.PDB import PDBIO, PDBParser, ShrakeRupley, StructureBuilder
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.DSSP import DSSP
_PDB_Parser = PDBParser()
SR = ShrakeRupley(
    probe_radius = 1.40,
    n_points = 100,
)

import druglib
from druglib.ops.dssp import DSSP_bin
from druglib.ops.msms import MSMS_bin
from druglib.alerts import check_inf_nan_np
from ..torch_utils import np_repeat
from . import protein_constants as pc
from .prot_math import extract_chi_and_template, to_pos14
from ..bio_utils.select_pocket import (
    select_bs_any, select_bs_atoms, select_bs_centroid,
)


@dataclass(frozen = True)
class Protein:
    """Protein structure representation"""
    name: str
    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    # [num_res, num_atom_type, 3], num_atom_type sets to 37 here
    # do not allow unknown atom
    atom_positions: np.ndarray

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices
    chain_index: np.ndarray # [num_res]

    # Optional remark about the protein.
    # Included as a comment in output PDB files
    remark: Optional[str] = None

    # atom property: every atom property
    # [num_res, num_atom_type], like atom-level SASA, solvent accessible surface area
    atom_prop: Optional[OrderedDict] = None

    # residue property: every residue property
    residue_prop: Optional[OrderedDict] = None  # [num_res, ], like residue-level SASA

    ####################### the below different from residue and atom representation
    # another representation about protein surface
    surface: Optional[np.ndarray] = None

    # surface property
    surface_prop: Optional[OrderedDict] = None

    # mask protein to get pocket residue
    # TODO: if neccessary, set attr pocket_mask to :obj:`complex` and remove this attr
    pocket_mask: Optional[np.ndarray] = None # [num_res, ]

    # pocket property
    pocket_prop: Optional[OrderedDict] = None

    # Bio object
    model: Optional[Bio.PDB.Model.Model] = None

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > pc.PDB_MAX_CHAINS:
            raise ValueError(f'Cannot build an instance with more than {pc.PDB_MAX_CHAINS} chains '
                             f'because these cannot be written to PDB format.')

    def query_atom_pos(
            self,
            atom_name:str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query atom position.
        Returns:
            queried atom position with the shape (num_res, 3).
            atom mask with the shape (num_res, ), which indicates
                the quered atom whether exists in the residue.
        """
        assert atom_name in pc.atom_types, \
            f"query atom `{atom_name}` is not in the predefined types."
        idx = pc.atom_order[atom_name]
        return self.atom_positions[:, idx, :], self.atom_mask[:, idx]

    def query_ca(self) -> Tuple[np.ndarray, np.ndarray]:
        atom_pos, mask = self.query_atom_pos('CA')
        if atom_pos.shape[0] != mask.sum():
            raise ValueError(
                "There are residues without `CA` atom."
            )
        return atom_pos, mask

    def query_n(self) -> Tuple[np.ndarray, np.ndarray]:
        atom_pos, mask = self.query_atom_pos('N')
        if atom_pos.shape[0] != mask.sum():
            raise ValueError(
                "There are residues without `N` atom."
            )
        return atom_pos, mask

    def query_c(self) -> Tuple[np.ndarray, np.ndarray]:
        atom_pos, mask = self.query_atom_pos('C')
        if atom_pos.shape[0] != mask.sum():
            raise ValueError(
                "There are residues without `C` atom."
            )
        return atom_pos, mask

    def query_o(self) -> Tuple[np.ndarray, np.ndarray]:
        atom_pos, mask = self.query_atom_pos('O')
        if atom_pos.shape[0] != mask.sum():
            raise ValueError(
                "There are residues without `O` atom."
            )
        return atom_pos, mask

    def query_atomsgroup(
            self,
            atoms: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query a group of atoms array"""
        group_ids = [pc.atom_order[atom] for atom in atoms]
        return self.atom_positions[:, group_ids, :], self.atom_mask[:, group_ids]

    def query_region(
            self,
            ref_coordinates: Union[np.ndarray, Tensor],
            selection_mode: Union[str, List[str]] = 'any',
            radius: Union[float, int] = 10.,
            max_neighbors: Optional[int] = None,
            return_mask: bool = False,
    ) -> Union['Protein', Tuple['Protein', np.ndarray]]:
        """
        Query a group of atoms in the specified region by given the
            region selection mode (centroid, `atom`, any), reference coordinates,
            cutoff radius.
        1. centroid mode means that selection criterion is the distance
            between the reference coordinates and the residues' centroid, then
            using the cutoff to select the residues forming the region.
        2. `atom` mode, consisting of atomtypes37, means that selection criterion
            is the distance between the reference coordinates and specific atomtypes,
            then using the cutoff to select the residues forming the region.
            Note that for every residue, if all atom type-reference coordinate
                minimum distances is within cutoff, then the residue will be selected.
            `All` rule compared with `any` rule
        3. any mode means that selection criterion is the distance between the reference
            coordinates and any one atom in the residue within the cutoff, then select the one.
        Args:
            ref_coordinates: np.ndarray or Tensor. Shape (N, 3) or (3, ) could be allowed
            selection_mode: str or list of str. if str type, 'centroid', `any`, any type
                in atomtypes or atom element (C, N, O, S), and the list of atomtypes or atom
                element will be allowed. Defaults to 'any'.
            radius: int or float. The cutoff radius. Defaults  to 10 A. Angstrom as the unit.
            max_neighbors: int, optional. The maximum number of residues selected.
                If exceeding, choose the nearest 'max_neighbors' ones.
                Defaults to None, indicating no upper bound.
            return_mask: bool. Whether return the residue mask that select the residues from the
                original :obj:`Protein`. Defaults to False.
        Returns:
            :obj:`Protein`: a new object `Protein`.
            residue_mask: np.ndarray.
        """
        assert isinstance(ref_coordinates, (np.ndarray, Tensor))
        if isinstance(ref_coordinates, np.ndarray):
            ref_coordinates = torch.from_numpy(ref_coordinates)
        if ref_coordinates.shape == (3, ):
            ref_coordinates = ref_coordinates.view(1, -1)
        if not (len(ref_coordinates.shape) == 2 and ref_coordinates.shape[-1] == 3):
            raise ValueError("Only Shape (N, 3) or (3,) could be allowed.")

        assert druglib.is_str(selection_mode) or druglib.is_list_of(selection_mode, str)
        atom_positions = self.atom_positions
        if selection_mode == 'centroid' or selection_mode == 'any':
            mode = selection_mode
            if mode == 'centroid':
                atom_positions = self.center_of_mass()
                select_fn = select_bs_centroid
            elif mode == 'any':
                select_fn = select_bs_any
        else:
            if druglib.is_str(selection_mode):
                selection_mode = [selection_mode]
            mode = []
            for i in selection_mode:
                assert i in pc.atom_types or \
                       i in pc.atom_element
                if i in pc.atom_element:
                    mode.extend([pc.atom_order[a] for a in \
                    pc.atom_types if a[0] == i])
                elif i in pc.atom_types:
                    mode.append(pc.atom_order[i])
            mode = sorted(list(set(mode)))
            select_fn = select_bs_atoms

        res_mask = select_fn( # noqa
            lig_pos = ref_coordinates,
            all_atom_positions = torch.from_numpy(atom_positions),
            all_atom_mask = torch.from_numpy(self.atom_mask),
            atoms_id = mode,
            cutoff = radius,
            max_neighbors = max_neighbors,
        )
        res_mask = res_mask.clone().cpu().numpy()

        # Final target, get the residue mask to select the that meets the cutoff radius.
        prot = self.apply_mask(res_mask)

        if return_mask:
            return prot, res_mask
        else:
            return prot

    def center_of_mass(self) -> np.ndarray:
        """
        Get every residue center of mass.
        Returns:
            residue center of mass. Shape (num_res ,3)
        """
        # get [num_res, num_atom_type] weights
        atom_weights = np_repeat(pc.atom_types_weight, self.aatype.shape[0])
        # reshape to [num_res, num_atom_type, 1] so allow multiplication and division
        atom_weights = atom_weights.reshape(atom_weights.shape + (1,))
        weighted_pos = self.atom_positions * atom_weights
        # mask atoms which do not exist
        aw_masked = atom_weights * self.atom_mask.reshape(self.atom_mask.shape + (1,))
        # get [num_res ,3]
        cm = weighted_pos.sum(axis = 1) / aw_masked.sum(axis = 1)
        return cm

    def num_atoms(self) -> int:
        return int(self.atom_mask.sum())

    def num_res(self) -> int:
        return self.atom_mask.shape[0]

    def to_pos14(
            self, consider_missing_atoms: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        """Consider missing atoms if set to True"""
        atom_positions, restype_atom14_mask = to_pos14(
            self.aatype,
            self.atom_positions,
            self.atom_mask if consider_missing_atoms else None,
        )
        return atom_positions, restype_atom14_mask

    def to_dense(self) -> OrderedDict:
        """
        Dense the residue level protein representation to atom level representation.
        Note that the :obj:`Protein` is sparse residue level representation, so it can
            be unable to output another :obj:`Protein` with dense representation, mainly
            non-functional or unable to use any methods of :obj:`Protein`.
        Returns:
            OrderedDict Entry:
                1. atom_positions: atom positions [num_atoms, 3]
                2. atom_label: atom37 label [num_atoms]
                3. b_factors: atom b factors [num_atoms]
                4. ptr: slice the atoms [num_atoms, ...] to residue level [num_residues, ...]
                    function atom_positions[ptr[i]:ptr[i+1]] is atoms position of redsidue i
                    i \in {0, 1, ..., M - 1} for M residues  [num_residues + 1]
                5. batch； atom level residue label. If N atoms is belong to jth residue, then
                    batch[ptr[j]:ptr[j+1]] = [j, ..., j]
                6. aatype: atom level AA type [num_atoms]
                7. residue_index: the index of residue in PDB file at atom level as like to aatype [num_atoms]
                8. chain_index: the index of chains of the atoms [num_atoms]
                9. pocket_mask: indicate which atoms is in pocket [num_atoms]  Optional
        """
        atom_positions = self.atom_positions
        atom_mask = self.atom_mask
        collect = OrderedDict()
        collect['name'] = self.name
        # dense to [num_atoms, 3]
        atom_positions = atom_positions[atom_mask == 1]
        collect['atom_positions'] = np.float32(atom_positions)
        # get every atom label [num_atoms]
        atom_label = np_repeat(np.arange(atom_mask.shape[-1]), atom_mask.shape[0])
        atom_label = atom_label[atom_mask == 1]
        collect['atom_label'] = np.int32(atom_label)
        # dense to [num_atoms]
        b_factors = self.b_factors[atom_mask == 1]
        collect['b_factors'] = np.float32(b_factors)
        if self.atom_prop is not None:
            atom_prop = OrderedDict()
            for k, v in self.atom_prop.items():
                value = v[atom_mask == 1]
                atom_prop[k] = copy.deepcopy(value)
            collect['atom_prop'] = atom_prop

        # ptr for dense the array index, mapping every residue to atoms
        atoms = np.int32(atom_mask.sum(axis = 1))
        ptr = np.empty(atoms.shape[0] + 1)
        ptr[0] = 0
        ptr[1:] = atoms.cumsum(0)
        collect['ptr'] = ptr
        # get atom-wise residue label --- batch
        batch = [np.full((n, ), i) for i, n in enumerate(atoms)]
        batch = np.concatenate(batch, axis = 0)
        collect['batch'] = batch
        for k in ['aatype', 'residue_index', 'chain_index', 'pocket_mask']:
            # residue info dense to atom info
            _key = getattr(self, k, None)
            if _key is not None:
                collect[k] = _key[batch]
        if self.residue_prop is not None:
            residue_prop = OrderedDict()
            for k, v in self.residue_prop.items():
                residue_prop[k] = v[batch]
            collect['residue_prop'] = residue_prop

        return collect

    @classmethod
    def to_sparse(
            cls,
            dense: OrderedDict,
    ) -> 'Protein':
        """
        Sparse to dense or say it is :method:`to_dense` reverse method.
        dense: OrderedDict. see :method:`to_dense` for details.
        """
        # localization usgae
        atom_label = np.int32(dense['atom_label'])
        # localizatiion requirements
        atom_positions = np.float32(dense['atom_positions'])
        b_factors = np.float32(dense['b_factors'])
        # separate usage
        ptr = np.int32(dense['ptr'])
        # usage for check
        batch = np.int32(dense['batch'])
        # residue property
        aatype = np.int32(dense['aatype'])
        residue_index = np.int32(dense['residue_index'])
        chain_index = np.int32(dense['chain_index'])
        pocket_mask = dense.get('pocket_mask', None)
        atom_prop = copy.deepcopy(dense.get('atom_prop', None))
        residue_prop = copy.deepcopy(dense.get('residue_prop', None))
        if atom_prop is not None:
            atomp = defaultdict(list)
        if residue_prop is not None:
            resp = defaultdict(list)

        def _slice_single_res(idx) -> tuple:
            """
            Separate the idx-th residue from the dense data
            """
            start, end = ptr[idx], ptr[idx + 1]
            # check batch is unique
            flag = np.unique(batch[start:end])
            if len(flag) != 1:
                raise ValueError(
                    f"Dense to Sparse Checking: When separating the {idx}-th residue, "
                    "find atoms are from different residue."
                )
            # get slicer
            select = atom_label[start:end]
            # initialize a residue representation for position
            residue_pos = np.zeros((pc.atom_type_num, 3), dtype = np.float32)
            residue_pos[select] = atom_positions[start:end]
            # initialize a residue representation for b-factor
            b_factor = np.zeros(pc.atom_type_num, dtype = np.float32)
            b_factor[select] = b_factors[start:end]
            # initialize a residue atom mask
            atom_mask = np.zeros(pc.atom_type_num, dtype = np.int32)
            atom_mask[select] = 1
            # initialize a atom_prop
            if atom_prop is not None:
                for k, v in atom_prop.items():
                    _v = np.zeros(pc.atom_type_num, dtype = v.dtype)
                    _v[select] = v[start:end]
                    atomp[k].append(_v)

            residue_prop_list = []
            for k in [aatype, residue_index, chain_index, pocket_mask]:
                if k is not None:
                    flag = np.unique(k[start:end])
                    if len(flag) != 1:
                        raise ValueError(
                            f"Dense to Sparse Checking: When separating the {idx}-th residue, "
                            "find atoms are from different residue."
                        )
                    residue_prop_list.append(int(k[start]))
                else:
                    residue_prop_list.append(None)

            if residue_prop is not None:
                for k, v in residue_prop.items():
                    flag = np.unique(v[start:end])
                    if len(flag) != 1:
                        raise ValueError(
                            f"Dense to Sparse Checking: When separating the {idx}-th residue, "
                            "find atoms are from different residue in the residue property."
                        )
                    resp[k].append(v[start])

            return (residue_pos, b_factor, atom_mask, residue_prop_list)

        residues = [_slice_single_res(i) for i in range(len(ptr) - 1)]
        atom_positions, b_factors, atom_mask, props = tuple(map(list, zip(*residues)))
        aatype, residue_index, chain_index, pocket_mask = tuple(map(list, zip(*props)))
        if atom_prop is not None:
            atom_prop = OrderedDict()
            for k, v in atomp.items():
                atom_prop[k] = np.stack(v, axis = 0)
        if residue_prop is not None:
            residue_prop = OrderedDict()
            for k, v in resp.items():
                residue_prop[k] = np.array(v)

        return cls(
            name = dense['name'],
            model = dense.get('model', None),
            atom_positions = np.stack(atom_positions, axis = 0),
            atom_mask = np.stack(atom_mask,  axis = 0),
            aatype = np.array(aatype),
            residue_index = np.array(residue_index),
            chain_index = np.array(chain_index),
            b_factors = np.stack(b_factors,  axis = 0),
            pocket_mask = np.array(pocket_mask) \
                if pocket_mask[0] is not None else None,
            atom_prop = atom_prop,
            residue_prop = residue_prop,
        )

    def apply_mask(self, res_mask):
        # TODO: use magic method __getitem__
        new_atom_prop = OrderedDict()
        new_residue_prop = OrderedDict()
        if self.atom_prop is not None:
            for k, v in self.atom_prop.items():
                new_atom_prop[k] = v[res_mask]
        if self.residue_prop is not None:
            for k, v in self.residue_prop.items():
                new_residue_prop[k] = v[res_mask]

        fragment = Protein(
            name = self.name,
            atom_positions = self.atom_positions[res_mask],
            atom_mask = self.atom_mask[res_mask],
            aatype = self.aatype[res_mask],
            residue_index = self.residue_index[res_mask],
            chain_index = self.chain_index[res_mask],
            b_factors = self.b_factors[res_mask],
            atom_prop = new_atom_prop,
            residue_prop = new_residue_prop,
        )

        return fragment

    def pos_update(
            self,
            new_pos: Union[np.ndarray, Tensor],
            model: Optional[Bio.PDB.Model.Model] = None,
            enable_build_model: bool = False,
    ) -> 'Protein':
        """
        new_pos: pos37. Shape (N_res, 37, 3) or pos14 Shape (N_res, 14, 3)
        """
        if isinstance(new_pos, Tensor):
            new_pos = new_pos.cpu().numpy()
        new_shape = new_pos.shape
        assert len(new_shape) == 3 and new_shape[1] in [14, 37], \
            f'New position shape must be (N_res, 37 | 14, 3), but got shape {new_shape}'
        is_pos37 = False
        if new_pos.shape[1] == 37:
            is_pos37 = True
        build_model = False
        if model is None and enable_build_model:
            build_model = True
        num_res = self.num_res()

        if build_model:
            pos14 = new_pos
            atoms37_to_atoms14 = pc.atoms37_to_atoms14_mapper[self.aatype]
            atom14_gt_exists = self.atom_mask[np.arange(num_res).reshape((-1, 1)), atoms37_to_atoms14]
            atom14_mask = pc.restype_atom14_mask[self.aatype]
            atom14_gt_exists = atom14_gt_exists * atom14_mask

            if is_pos37:
                pos14 = new_pos[np.arange(num_res).reshape((-1, 1)), atoms37_to_atoms14]
                pos14 = pos14 * atom14_gt_exists.reshape((-1, 14, 1))

            model = self.pdb_from_pos14(
                pos14, self.aatype, self.b_factors, atom14_gt_exists,
                self.residue_index, self.chain_index, None)

        # C-terminal or N-terminal loss if pos14 to pos37
        atom_mask = self.atom_mask
        if not is_pos37:
            atoms14_to_atoms37 = pc.atoms14_to_atoms37_mapper[self.aatype]
            new_pos = new_pos[np.arange(num_res).reshape((-1, 1)), atoms14_to_atoms37]
            new_pos = new_pos * atom_mask.reshape((-1, 37, 1))

        return Protein(
            name = self.name,
            atom_positions = new_pos,
            aatype = self.aatype,
            atom_mask = atom_mask,
            residue_index = self.residue_index,
            b_factors = self.b_factors,
            chain_index = self.chain_index,
            remark = self.remark,
            atom_prop = self.atom_prop,
            residue_prop = self.residue_prop,
            surface = self.surface,
            surface_prop = self.surface_prop,
            pocket_mask = self.pocket_mask,
            pocket_prop = self.pocket_prop,
            model = model,
        )

    def extract_chi_and_template(
            self,
            return_radian: bool = False,
    ) -> ed:
        atom_positions, restype_atom14_mask = self.to_pos14()
        return extract_chi_and_template(
            self.aatype,
            atom_positions,
            restype_atom14_mask,
            return_radian = return_radian
        )

    @staticmethod
    def pdb_from_pos14(
            pos14: np.ndarray,
            sequence: np.ndarray,
            b_factors: Optional[np.ndarray] = None,
            mask14: Optional[np.ndarray] = None,
            residue_index: Optional[np.ndarray] = None,
            chain_index: Optional[np.ndarray] = None,
            save_path: Optional[str] = None,
            model: int = 0,
    ) -> Optional[Bio.PDB.Model.Model]:
        """
        Save pos14 protein atoms array to pdb file.
        Args:
            pos14: np.ndarray. Shape [num_residues, 14, 3].
                The atoms type must match the `pc.restype_name_to_atom14_names`.
            sequence: AAtype as the np.ndarray. Shape [num_residues, ].
                The residue tupe must be match `pc.restype_order`.
            b_factors: np.ndarray, optional. Shape [num_residues, ].
                Crystal structure b-factor or pLDDT from alphafold2 model.
                If unspecified, defaults to ones.
            residue_index: np.ndarray, optional. Shape [num_residues, ]
                If unspecified, defaults to 0-num_residues.
            mask14: np.ndarray, optional. Shape [num_residues, 14].
                Match to sequence, indicates the real (1) or placeholder (0)
                    residue existing atoms.
            chain_index: np.ndarray, optional. Shape [num_residues, ].
                Indicates the residue belong to which chain, so we can support multimer pdb build.
                If unspecified, defaults to ones (->A).
        Fix: use mask14 to recognize missing atoms in specified residues.
        """
        if save_path is not None and not druglib.is_str(save_path):
            raise TypeError(f'input args `save_path` must be string type or None, but got {type(save_path)}')
        n_res = pos14.shape[0]

        if b_factors is None:
            b_factors = np.ones(n_res, dtype = np.float32)

        if mask14 is None:
            mask14 = pc.restype_atom14_mask[sequence]
            mask14 = mask14.astype('bool')
        mask = (np.sum(mask14, axis = -1) > 0)

        if chain_index is None:
            chain_index = np.ones(n_res, dtype = np.int32)
        id_chain_mapping = np.asarray(list(string.ascii_uppercase))
        chain_index = id_chain_mapping[chain_index]

        # analysis disordered residue and count them
        disordered_count = {}
        disordered_mapping = string.ascii_uppercase
        if residue_index is None:
            residue_index = np.arange(n_res)
        for chain_idx, resid_idx in zip(chain_index, residue_index):
            cha_res_id = f'{str(chain_idx)}{int(resid_idx)}'
            if cha_res_id not in disordered_count:
                disordered_count[cha_res_id] = 1
            else:
                disordered_count[cha_res_id] += 1
        disordered_count = {k: disordered_mapping[:v] for k, v in disordered_count.items() if v > 1}

        builder = StructureBuilder.StructureBuilder()
        builder.init_structure(0)
        builder.init_model(model)
        init_chain = None
        for i, (aa_idx, p_res, mr, b, m_res, chain_idx, resid_idx) in enumerate(
                zip(sequence, pos14, mask14, b_factors, mask, chain_index, residue_index)
        ):
            if not m_res:
                continue

            if aa_idx == 21:
                continue

            if chain_idx != init_chain:
                init_chain = chain_idx
                builder.init_chain(init_chain)
                builder.init_seg('    ')

            three = pc.restype_1to3[pc.restypes_with_x[aa_idx]]
            insert_code = " "
            cha_res_id = f'{str(chain_idx)}{int(resid_idx)}'
            if cha_res_id in disordered_count:
                insert_code = disordered_count[cha_res_id][0]
                disordered_count[cha_res_id] = disordered_count[cha_res_id][1:]
            builder.init_residue(three, " ", int(resid_idx), icode = insert_code)

            for j, (atom_name,) in enumerate(
                    zip(pc.restype_name_to_atom14_names[three])
            ):
                if (len(atom_name) > 0) and mr[j]:
                    builder.init_atom(
                        atom_name, p_res[j].tolist(), b, 1.0, ' ',
                        atom_name.join([" ", " "]), element = atom_name[0])

        structure = builder.get_structure()
        if save_path is None:
            return structure

        io = PDBIO()
        io.set_structure(structure)
        druglib.mkdir_or_exists(pathlib.Path(save_path).parent)
        io.save(save_path)

    def to_pdb(self) -> str: return to_pdb(self)

def int_id_to_str_id(num: int) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.
    Args:
        num: A positive integer.

    Returns:
        A string that encodes the positive integer using reverse spreadsheet style,
            naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
            usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
        raise ValueError(f'Only positive integers allowed, got {num}.')

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord('A')))
        num = num // 26 - 1
    return ''.join(output)

def to_pdb(
        protein: Protein,
        model: Optional[int] = None,
        add_end: bool = True,
) -> str:
    """
    Convert the :obj:`Protein` to pdb file to build the protein.
    Mainly for the :obj:`Protein` visualization.
    In default, write the single protein; When multiple protein (trajectories)
        are written to pdb file, turn args `model` to int, and set `add_end` to True.
    Returns:
        PDBBlock: str
    """
    restypes = pc.restypes_with_x
    restype1to3 = pc.restype_1to3
    atom_types = pc.atom_types

    atom_positions = protein.atom_positions
    atom_mask = protein.atom_mask
    aatype = protein.aatype
    residue_index_pdb = protein.residue_index
    chain_ids = protein.chain_index
    bfactors = protein.b_factors

    if np.any(aatype > pc.restype_num):
        raise ValueError("Invalid residue type found, only allow natural AA.")

    lines = []
    remark = protein.remark
    if remark is not None:
        lines.append(remark)
    elif (model == None) or (model == 1):
        lines.append('REMARK   1 CREATED WITH MDLDruglib %s, %s' % (druglib.__version__, str(date.today())))

    id_chain_mapping = string.ascii_uppercase
    # analysis disordered residue and count them
    disordered_count = {}
    for i, resid in enumerate(residue_index_pdb):
        chain_tag = 'A'
        if chain_ids is not None:
            # chain_tag = id_chain_mapping[chain_ids[i]]
            chain_tag = int_id_to_str_id(chain_ids[i] + 1)        

        if resid not in disordered_count:
            disordered_count[(chain_tag, resid)] = 1
        else:
            disordered_count[(chain_tag, resid)] += 1
    disordered_count = {k: id_chain_mapping[:v] for k, v in disordered_count.items() if v > 1}

    N = aatype.shape[0]
    atom_idx = 1
    for i in range(N):
        chain_tag = 'A'
        if chain_ids is not None:
            # chain_tag = id_chain_mapping[chain_ids[i]]
            chain_tag = int_id_to_str_id(chain_ids[i] + 1)            

        resn3 = restype1to3.get(restypes[aatype[i]], 'UNK')

        insert_code = ''
        residue_ind = residue_index_pdb[i]
        ch_res_tuple = (chain_tag, residue_ind)
        if ch_res_tuple in disordered_count:
            insert_code = disordered_count[ch_res_tuple][0]
            disordered_count[ch_res_tuple] = disordered_count[ch_res_tuple][1:]

        for atn, pos, mask, bf in zip(
            atom_types, atom_positions[i], atom_mask[i], bfactors[i]
        ):
            if mask < 0.5:
                continue
            atom_tag = 'ATOM'
            name = atn if len(atn) == 4 else f' {atn}'
            alt_loc = ''
            occupancy = 1.00
            element = atn[0]
            charge = ''

            atn_line = (
                f"{atom_tag:<6}{atom_idx:>5} {name:<4}{alt_loc:>1}"
                f"{resn3:>3} {chain_tag:>1}"
                f"{residue_ind:>4}{insert_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{bf:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            lines.append(atn_line)
            atom_idx += 1

        should_terminate = (i == N - 1)
        if chain_ids is not None:
            if i != N - 1 and chain_ids[i] != chain_ids[i + 1]:
                should_terminate = True

        if should_terminate:
            chain_end = 'TER'
            chain_termination_line = (
                f"{chain_end:<6}{atom_idx:>5}      "
                f"{resn3:>3} "
                f"{chain_tag:>1}{residue_ind:>4}"
            )
            lines.append(chain_termination_line)
            atom_idx += 1

    if model is not None:
        lines.append('ENDMDL')
    if add_end:
        lines.append('END')

    lines = [line.ljust(80) for line in lines]
    return '\n'.join(lines) + '\n'

modified_mapping = {
    "MSE": "MET", "MLY": "LYS", "FME": "MET", "HYP": "PRO",
    "TPO": "THR", "CSO": "CYS", "SEP": "SER", "M3L": "LYS",
    "HSK": "HIS", "SAC": "SER", "PCA": "GLU", "DAL": "ALA",
    "CME": "CYS", "CSD": "CYS", "OCS": "CYS", "DPR": "PRO",
    "B3K": "LYS", "ALY": "LYS", "YCM": "CYS", "MLZ": "LYS",
    "4BF": "TYR", "KCX": "LYS", "B3E": "GLU", "B3D": "ASP",
    "HZP": "PRO", "CSX": "CYS", "BAL": "ALA", "HIC": "HIS",
    "DBZ": "ALA", "DCY": "CYS", "DVA": "VAL", "NLE": "LEU",
    "SMC": "CYS", "AGM": "ARG", "B3A": "ALA", "DAS": "ASP",
    "DLY": "LYS", "DSN": "SER", "DTH": "THR", "GL3": "GLY",
    "HY3": "PRO", "LLP": "LYS", "MGN": "GLN", "MHS": "HIS",
    "TRQ": "TRP", "B3Y": "TYR", "PHI": "PHE", "PTR": "TYR",
    "TYS": "TYR", "IAS": "ASP", "GPL": "LYS", "KYN": "TRP",
    "SEC": "CYS",
}

def pdb_parser(
        pdb_path: str,
        chain_id: Optional[str] = None,
        use_residuedepth: bool = False,
        use_ss: bool = False,
        calc_sasa: bool = False,
        name: Optional[str] = None,
) -> Protein:
    """
    Parse PDB file to constructs a Protein object.
    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.
    Args:
        pdb_path: str. The pdb file
        chain_id: If None, then the pdb file must contain a single chain (which
            will be parsed). If chain_id is specified (e.g. A), then only that chain
            is parsed.
        use_residuedepth: bool. Use Michel Sanner’s MSMS program for the surface calculation,
            And residue depth is the average distance of the atoms of a residue from the
            solvent accessible surface. Defaults to False.
        use_ss: bool. Calculate secondary structure and accessibility. Defaults to False.
        calc_sasa: bool. Calculate the atom and residue level SASA. Defaults to False.
    Returns:
        A new `Protein` parsed from the pdb file.
    """
    assert osp.exists(pdb_path), f"PDB file does not exist from {pdb_path}."
    if use_residuedepth:
        use_residuedepth = False if MSMS_bin is None else True
    if use_ss:
        use_ss = False if DSSP_bin is None else True

    pdbid = osp.basename(pdb_path) if name is None else str(name)
    structure = _PDB_Parser.get_structure("Random Name", pdb_path)
    models = list(structure.get_models())
    if len(models) != 1:
        warnings.warn(
            f"Only single model PDBs are supported. Found {len(models)} models."
            "Just use the first one."
        )
    model = models[0]

    if calc_sasa:
        SR.compute(model, level = 'R')
        SR.compute(model, level = 'A')

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []
    sasa_atoms = []
    sasa_res = []
    # residue level property
    if use_residuedepth:
        res_rd, ca_rd = [], []
        rd = ResidueDepth(model, msms_exec = MSMS_bin)
    if use_ss:
        ss, rasa, phi, psi = [], [], [], []
        dssp = DSSP(model, pdb_path, dssp = DSSP_bin)
    atom_prop = OrderedDict()
    res_prop = OrderedDict()

    for _chain_id, chain in enumerate(model):
        if(chain_id is not None and chain.id != chain_id):
            continue
        for res in chain:
            # check point
            # excluding the water and other nonstandard AA and nonstandard nuclear acids, solvate etc
            if res.id[0] != ' ':
                continue
            if use_ss and (chain.id, res.id) not in dssp:
                continue

            if calc_sasa:
                res_sasa = res.sasa
                assert check_inf_nan_np(res_sasa), \
                    f'`residue_sasa` is inf or nan from {pdb_path}.'
            
            resn3 = res.resname
            if resn3 in modified_mapping:
                resn3 = modified_mapping[resn3]

            res_shortname = pc.restype_3to1.get(resn3, "X")
            if res_shortname == "X":
                continue

            restype_idx = pc.restype_order.get(
                res_shortname, pc.restype_num
            )
            pos = np.zeros((pc.atom_type_num, 3))
            mask = np.zeros((pc.atom_type_num,))
            res_b_factors = np.zeros((pc.atom_type_num,))
            atoms_sasa = np.zeros((pc.atom_type_num,))
            for atom in res:
                # check point
                if atom.name not in pc.atom_types:
                    # ignore the H atom and focus on the heavy atoms
                    continue
                bfactor = atom.bfactor
                assert check_inf_nan_np(bfactor), \
                    f'`bfactor` is inf or nan from {pdb_path}.'

                pos[pc.atom_order[atom.name]] = atom.coord
                mask[pc.atom_order[atom.name]] = 1.0
                res_b_factors[
                    pc.atom_order[atom.name]
                ] = bfactor

                if calc_sasa:
                    atom_sasa = atom.sasa
                    assert check_inf_nan_np(atom_sasa), \
                        f'`atom_sasa` is inf or nan from {pdb_path}.'
                    atoms_sasa[
                        pc.atom_order[atom.name]
                    ] = atom_sasa

            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue

            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            # contains disordered, check by red.id[2]
            # we allow the disordered residue
            residue_index.append(res.id[1])
            chain_ids.append(_chain_id) # chain id use pseudo id, NOT PDB chain id
            b_factors.append(res_b_factors)

            if calc_sasa:
                sasa_atoms.append(atoms_sasa)
                sasa_res.append(res_sasa)
            if use_residuedepth:
                _res_rd, _ca_rd = rd[chain.id, res.id]
                res_rd.append(_res_rd)
                ca_rd.append(_ca_rd)
            if use_ss:
                _res_dssp = dssp[(chain.id, res.id)]
                ss.append(pc.ss_label.index(_res_dssp[2]))
                rasa.append(_res_dssp[3])
                phi.append(_res_dssp[4])
                psi.append(_res_dssp[5])

    # TODO: more consideration about this error
    if len(aatype) < 10:
        raise ValueError(f'Parsed protein file from {pdb_path} has less than 10 residues.')

    if calc_sasa:
        atom_prop['sasa'] = np.array(sasa_atoms)
        res_prop['sasa'] = np.array(sasa_res)
    if use_residuedepth:
        res_prop['res_rd'] = np.array(res_rd, dtype = float)
        assert check_inf_nan_np(res_prop['res_rd']), \
            f'`residue depth` of residues is inf or nan from {pdb_path}.'
        res_prop['ca_rd'] = np.array(ca_rd, dtype = float)
        assert check_inf_nan_np(res_prop['res_rd']) and \
               check_inf_nan_np(res_prop['ca_rd']), \
            f'`residue depth` is inf or nan from {pdb_path}.'
    if use_ss:
        res_prop['ss'] = np.array(ss, dtype = int)
        res_prop['rasa'] = np.array(rasa, dtype = float)
        res_prop['phi'] = np.array(phi, dtype = float)
        res_prop['psi'] = np.array(psi, dtype = float)
        assert check_inf_nan_np(res_prop['ss']) and \
               check_inf_nan_np(res_prop['rasa']) and \
               check_inf_nan_np(res_prop['phi']) and \
               check_inf_nan_np(res_prop['psi']), \
            f'secondary structure feature is inf or nan from {pdb_path}.'

    return Protein(
        name = pdbid,
        model = model,
        atom_positions = np.array(atom_positions),
        atom_mask = np.array(atom_mask),
        aatype = np.array(aatype),
        residue_index = np.array(residue_index),
        chain_index = np.array(chain_ids),
        b_factors = np.array(b_factors),
        atom_prop = atom_prop,
        residue_prop = res_prop,
    )
