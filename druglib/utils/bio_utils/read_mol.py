# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, Union
import os.path as osp
from rdkit import Chem
from .compute_mol_charges import compute_mol_charges
from .pdbqt_utils import pdbqt2pdbblock
from .conformer_utils import fast_generate_conformers_onebyone


def read_mol(
        mol_file: Union[str, Chem.rdchem.Mol],
        sanitize: bool = False,
        calc_charges: bool = False,
        remove_hs: bool = False,
        assign_chirality: bool = False,
        emb_multiple_3d: Optional[int] = None,
) -> Optional[Chem.rdchem.Mol]:
    """
    Read a molecule file in the format of .sdf, .mol2, .pdbqt or .pdb.
    Args:
        mol_file: str or Mol. Molecule file path or rdkit Mol object.
        sanitize: bool, optional. rdkit sanitize molecule.
            Default to False.
        calc_charges: bool, optional. If True, add Gasteiger charges.
            Default to False.
            Note that when calculating charges, the mol must be sanitized.
        remove_hs: bool, optional. If True, remove the hydrogens.
            Default to False.
        assign_chirality: bool, optional. If True, inference the chirality.
            Default to False.
        emb_multiple_3d: int, optional. If int, generate multiple conformers for mol
    Returns:
        molecule: Chem.rdchem.Mol or None.
    """
    if not isinstance(mol_file, Chem.rdchem.Mol):
        assert osp.exists(mol_file), f"Ligand file does not exist from {mol_file}."

    if isinstance(mol_file, Chem.rdchem.Mol):
        # Here we allow the valid mol input
        mol = mol_file
    elif mol_file.endswith('.sdf'):
        mols = Chem.SDMolSupplier(
            mol_file,
            sanitize = False,
            removeHs = False
        )
        # Note that this requires input a single molecule sdf file
        # if file saves multiply molecules, it is dangerous for execute
        # the next part.
        mol = mols[0]
    elif mol_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(
            mol_file,
            sanitize = False,
            removeHs = False,
        )
    elif mol_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(
            mol_file,
            sanitize = False,
            removeHs = False,
        )
    elif mol_file.endswith('.pdbqt'):
        pdbblock = pdbqt2pdbblock(mol_file)
        mol = Chem.MolFromPDBBlock(
            pdbblock,
            sanitize = False,
            removeHs = False,
        )
    else:
        raise ValueError("Current supported mol files include sdf, mol2, pdbqt, pdb, "
                         f"but got {mol_file.split('.')[-1]}")

    Chem.GetSymmSSSR(mol)

    if emb_multiple_3d is not None:
        assert isinstance(emb_multiple_3d, int) and emb_multiple_3d > 0
        mol = fast_generate_conformers_onebyone(
            mol, num_confs = emb_multiple_3d,
            force_field = 'MMFF94s',
        )

    try:
        if sanitize: Chem.SanitizeMol(mol)

        if calc_charges:
            try:
                compute_mol_charges(mol)
            except RuntimeError:
               pass

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize = sanitize)
    except:
        return None

    if assign_chirality:
        Chem.AssignStereochemistryFrom3D(mol)

    return mol
