# Copyright (c) MDLDrugLib. All rights reserved.
import logging
from typing import Optional
import os.path as osp
from rdkit import Chem
from druglib.utils.obj.protein import pdb_parser
from druglib.utils.obj.ligand import ligand_parser
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadLigand:
    """
    Load an :obj:`Ligand3D` from SDF file or mol2 file.
    Saved in the dict data 'ligand' from the 'ligand_file'
    Saved the 'ligand_file'.
    """
    supported_ext = ['.sdf', '.mol2']
    def __init__(
            self,
            sanitize: bool = True,
            calc_charges: bool = True,
            remove_hs: bool = False,
            assign_chirality: bool = True,
            allow_genconf: bool = True,
            emb_multiple_3d: Optional[int] = None,
            only_return_mol: bool = False,
    ):
        self.sanitize = sanitize
        self.calc_charges = calc_charges
        self.remove_hs = remove_hs
        self.assign_chirality = assign_chirality
        self.allow_genconf = allow_genconf
        self.emb_multiple_3d = emb_multiple_3d
        self.only_return_mol = only_return_mol

    def __call__(self, data):
        ligand_file = data['ligand_file']
        if not isinstance(ligand_file, Chem.rdchem.Mol):
            # TODO: inference phase smiles parser is required
            dir_name, ext = osp.splitext(ligand_file)
            if ext not in self.supported_ext:
                raise ValueError(f'Only {",".join(self.supported_ext)} file supported, '
                                 f'but got {ligand_file}')
            if not osp.exists(ligand_file):
                ligand_file = dir_name + self.supported_ext[1 - self.supported_ext.index(ext)]
            if not osp.exists(ligand_file):
                raise FileExistsError(f'Any {",".join(self.supported_ext)} file for the {dir_name} file is not Found')
            # solve the issue: AttributeError: 'PosixPath' object has no attribute 'endswith' in read_mol
            ligand_file = str(ligand_file)
        try:
            ligand = ligand_parser(
                ligand_file,
                sanitize = self.sanitize,
                calc_charges = self.calc_charges,
                remove_hs = self.remove_hs,
                assign_chirality = self.assign_chirality,
                allow_genconf = self.allow_genconf,
                emb_multiple_3d = self.emb_multiple_3d,
                only_return_mol = self.only_return_mol,
            )
            if ligand is None and self.sanitize:
                logging.warning('sanitize mol failed and try to turn off sanitize.')
                ligand = ligand_parser(
                    ligand_file,
                    sanitize=False,
                    calc_charges=self.calc_charges,
                    remove_hs=self.remove_hs,
                    assign_chirality=self.assign_chirality,
                    allow_genconf=self.allow_genconf,
                    emb_multiple_3d = self.emb_multiple_3d,
                    only_return_mol=self.only_return_mol,
                )

            if ligand is None:
                raise RuntimeError(f'Ligand from {ligand_file} is None.')
        except Exception as e:
            if not isinstance(ligand_file, Chem.rdchem.Mol):
                ligand_file = dir_name + self.supported_ext[1 - self.supported_ext.index(ext)]
                ligand = ligand_parser(
                    ligand_file,
                    sanitize = self.sanitize,
                    calc_charges = self.calc_charges,
                    remove_hs = self.remove_hs,
                    assign_chirality = self.assign_chirality,
                    allow_genconf = self.allow_genconf,
                    emb_multiple_3d = self.emb_multiple_3d,
                    only_return_mol = self.only_return_mol,
                    name = data.get('lig_name', None),
                )
                if ligand is None and self.sanitize:
                    ligand = ligand_parser(
                        ligand_file,
                        sanitize=False,
                        calc_charges=self.calc_charges,
                        remove_hs=self.remove_hs,
                        assign_chirality=self.assign_chirality,
                        allow_genconf=self.allow_genconf,
                        emb_multiple_3d = self.emb_multiple_3d,
                        only_return_mol=self.only_return_mol,
                    )
            else:
                raise RuntimeError(f'Unvalid input rdkit.Chem.rdchem.Mol with error message:\n {e}')

        data['ligand'] = ligand

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'sanitize={self.sanitize}, '
                f'calc_charges={self.calc_charges}, '
                f'remove_hs={self.remove_hs}, '
                f'assign_chirality={self.assign_chirality}, '
                f'emb_multiple_3d={self.emb_multiple_3d}, '
                f'allow_genconf={self.allow_genconf}'
                f')')

@PIPELINES.register_module()
class LoadProtein:
    """
    Load an :obj:`Protein` from PDB file.
    Saved in the dict data 'protein' from the 'protein_file'
    Saved the 'protein_file'
    """
    def __init__(
            self,
            use_residuedepth: bool = False,
            use_ss: bool = False,
    ):
        self.use_residuedepth = use_residuedepth
        self.use_ss = use_ss

    def __call__(self, data):
        prot_file = data['protein_file']
        dir_name, ext = osp.splitext(prot_file)
        if ext != '.pdb':
            raise ValueError('Only pdb file supported, '
                             f'but got {prot_file}')
        if not osp.exists(prot_file):
            raise FileExistsError(f"PDB file `{prot_file}` is not Found")

        prot = pdb_parser(
            prot_file,
            use_residuedepth = self.use_residuedepth,
            use_ss = self.use_ss,
            name = data.get('prot_name', None),
        )
        data['protein'] = prot

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'use_residuedepth={self.use_residuedepth}, '
                f'use_ss={self.use_ss}, '
                f')')