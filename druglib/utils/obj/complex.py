# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Optional,
    Union,
    List,
    Tuple,
)
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import torch
from Bio import PDB
from rdkit import Chem

from .protein import Protein
from .ligand import Ligand3D


@dataclass()
class PLComplex:
    """Protein-Ligand Complex topology and geometry representation"""
    # protein representation
    # Full-length protein or truncated pocket from the original PDB file
    protein: Protein

    # Ligand representation
    ligand: Ligand3D

    # protein atom index   node matrix segments (or index) to get protein node matrix
    proten_index: Optional[np.ndarray] = None

    # ligand atom index    node matrix segments (or index) to get ligand node matrix
    ligand_index: Optional[np.ndarray] = None

    # complex representation (position concatenation)
    complex_pos: Optional[np.ndarray] = None

    # protein dense representation  atom-wise
    dense_protein: Optional[OrderedDict] = None

    # complex property
    complex_info: Optional[OrderedDict] = None

    def __post_init__(self):
        """
        Initialize the stacked ligand-protein positions,
            ligand atom index, protein atom index.
        Protein position: (N ,3)
        Ligand position: (M, 3)
           ---> Protein-ligand pair (N + M, 3)
        Protein atom index: in the format of index with shape (N, )
        Ligand atom idnex: in teh format of index with shape (M, )
        Note that only sparse protein can be used to concat the
            node feature with ligand node feature. We do not do
            this in the 'post_init', because node feature is well-defined
            in the data transform pipelines, followed with the protein featurizer
            and ligand featurizer.
        """
        complex_position, ligand_index, \
        protein_index, dense = self.paired(
            self.protein,
            self.ligand
        )
        self.proten_index = protein_index
        self.ligand_index = ligand_index
        self.complex_pos = complex_position
        self.dense_protein = dense

    @staticmethod
    def paired(
            protein: Protein,
            ligand: Ligand3D,
    ) -> tuple:
        """Protein-Ligand position pairing"""
        dense = protein.to_dense()
        complex_position = np.concatenate(
            [ligand.atom_positions, dense['atom_positions']],
            axis = 0
        )
        num_allatoms = ligand.numatoms + dense['atom_positions'].shape[0]
        ligand_index = np.arange(ligand.numatoms, dtype = int)
        protein_index = np.arange(ligand.numatoms, num_allatoms, dtype = int)
        assert complex_position.shape[0] == num_allatoms, \
        f'atom positions ({complex_position.shape[0]}) and number of atoms ({num_allatoms}) mismatch.'

        return complex_position, ligand_index, protein_index, dense

    def get_pocket_ligand_pair(
            self,
            ligand_coords: Union[str, torch.Tensor, np.ndarray] = 'centroid',
            selection_mode: Union[str, List[str]] = 'any',
            radius: Union[float, int] = 10.,
            max_neighbors: Optional[int] = None
    ) -> Tuple['PLComplex', torch.Tensor]:
        """
        Get the truncated pocket-ligand pair.
        Args:
            ligand_coords: str or np.ndarray. Get pocket residue
                given the reference coordinates.
            Only three modes are supported, 'centroid' or 'all',
                and other np.ndarray coordinates input.
                Defaults to 'centroid'
            See `utils.obj.protein` :func:query_region
                for details.
        Returns:
            truncate pocket around the ligand, pocket-ligand pair.
        """
        if isinstance(ligand_coords, (np.ndarray, torch.Tensor)):
            pass
        elif ligand_coords == 'centroid':
            ligand_coords = self.ligand.cm
        elif ligand_coords == 'all':
            ligand_coords = self.ligand.atom_positions
        else:
            raise ValueError("Input args `ligand_coords` only supported for 'centroid' or "
                             f"'all' and np.ndarray coordinates, but got `{ligand_coords}`")

        region, region_mask = self.protein.query_region(
            ref_coordinates = ligand_coords,
            selection_mode = selection_mode,
            radius = radius,
            max_neighbors = max_neighbors,
            return_mask = True,
        )

        complex_position, ligand_index, \
        protein_index, dense = self.paired(
            region, self.ligand
        )

        return PLComplex(
            protein = region,
            proten_index = protein_index,
            ligand = self.ligand,
            ligand_index = ligand_index,
            complex_pos = complex_position,
            dense_protein = dense,
        ), region_mask

    def pos_update(
            self,
            prot_new_pos: Union[np.ndarray, torch.Tensor],
            lig_new_pos: Union[np.ndarray, torch.Tensor],
            prot_model: Optional[PDB.Model.Model] = None,
            enable_build_prot_model: bool = False,
            lig_model: Optional[Chem.rdchem.Mol] = None,
            lig_hasRemovedHs: bool = False,
    ) -> 'PLComplex':
        ligand = self.ligand.pos_update(lig_new_pos, lig_model, lig_hasRemovedHs)
        protein = self.protein.pos_update(prot_new_pos, prot_model, enable_build_prot_model)
        complex_position, ligand_index, \
        protein_index, dense = self.paired(
            protein,
            self.ligand,
        )

        return PLComplex(
            protein = protein,
            proten_index = protein_index,
            ligand = ligand,
            ligand_index = ligand_index,
            complex_pos = complex_position,
            dense_protein = dense)

    def to_pdb(self) -> str:
        """
        Convert the :obj:`PLComplex` to pdb file to build the
            protein-ligand complex.
        Mainly for the :obj:`PLComplex` visualization.
        Returns:
            PDBBlock: str
        """
        protein_pdbblock = self.protein.to_pdb()
        if self.ligand.model is None:
            return protein_pdbblock

        plines = protein_pdbblock.splitlines()
        remark_cutpoint = 0
        for line in plines:
            if 'REMARK' not in line.rstrip():
                break
            remark_cutpoint += 1
        ligand_pdbblock = Chem.MolToPDBBlock(self.ligand.model)
        llines = ligand_pdbblock.splitlines()
        connect_cutpoint = 0
        for line in llines:
            if 'CONECT' in line.rstrip():
                break
            connect_cutpoint += 1
        complex_block = plines[:remark_cutpoint] + llines[:connect_cutpoint] + \
                        plines[remark_cutpoint:] + llines[connect_cutpoint:] + ['\n']

        return '\n'.join(complex_block)