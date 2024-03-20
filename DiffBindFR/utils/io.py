# Copyright (c) MDLDrugLib. All rights reserved.
import os, warnings, json
from typing import Optional, Union
import os.path as osp
from pathlib import Path
import numpy as np

from rdkit import Chem
from rdkit.Geometry import Point3D


class JSONEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)

def exists_or_assert(
        file: Union[str, Path],
        allow_raise_error: bool = True,
        allow_warning: bool = False,
) -> bool:
    if not osp.exists(file):
        if allow_raise_error:
            raise FileNotFoundError(f'{file} does not exist.')
        else:
            if allow_warning:
                warnings.warn(f'{file} is missing.')
            return False
    return True

def mkdir_or_exists(
        dirname: str,
        mode: int = 0o777,
) -> None:
    if dirname == '':
        return
    dirname = osp.expanduser(dirname)
    os.makedirs(dirname, mode = mode, exist_ok = True)
    return

def filename(
        file: str,
):
    basename = osp.basename(file)
    name = basename.split('.')[0]
    return name

def write_fasta(
        fasta: str,
        pdbid: str,
        sequence: str,
):
    fasta = osp.abspath(fasta)
    exists_or_assert(fasta)
    with open(fasta, 'w') as f:
        f.write('>' + pdbid + '\n')
        f.write(sequence)
    return fasta

def read_mol(
        mol_file: Union[str, Chem.rdchem.Mol],
) -> Optional[Chem.rdchem.Mol]:
    if not isinstance(mol_file, Chem.rdchem.Mol):
        assert osp.exists(mol_file), f"Ligand file does not exist from {mol_file}."

    if isinstance(mol_file, Chem.rdchem.Mol):
        mol = mol_file
    elif mol_file.endswith('.sdf'):
        mols = Chem.SDMolSupplier(
            mol_file,
            sanitize = False,
            removeHs = False
        )
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
    else:
        raise ValueError("Current supported mol files include sdf, mol2 and pdb, "
                         f"but got {mol_file.split('.')[-1]}")

    Chem.GetSymmSSSR(mol)
    return mol

def to_complex_block(
        pblock: str,
        lblock: str,
        output: Optional[str] = None
) -> str:
    """Encode complex pdb block"""
    plines = pblock.splitlines()
    remark_cutpoint = 0
    for line in plines:
        if 'REMARK' not in line.rstrip():
            break
        remark_cutpoint += 1
    llines = lblock.splitlines()
    connect_cutpoint = 0
    for line in llines:
        if 'CONECT' in line.rstrip():
            break
        connect_cutpoint += 1
    complex_block = plines[:remark_cutpoint] + llines[:connect_cutpoint] + \
                    plines[remark_cutpoint:] + llines[connect_cutpoint:] + ['\n']

    complex_block = '\n'.join(complex_block)

    if output is not None:
        Path(output).parent.mkdir(exist_ok = True, parents = True)
        with open(output, 'w') as fw:
            fw.write(complex_block)

    return complex_block


def read_molblock(
        pdb_file: str,
):
    """Decode complex pdb block to the ligand block"""
    pdb_file = str(pdb_file)
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    molblock = []
    for line in lines:
        if line.startswith('HETATM') or line.startswith('CONECT'):
            molblock.append(line)

    molblock.append('END')

    return ''.join(molblock)

def update_mol_pose(
        mol_seed: Chem.Mol,
        mol_focus: Chem.Mol,
) -> Chem.Mol:
    """Transfer the position info from mol_focus to mol_seed with the same topology"""
    assert Chem.MolToSmiles(mol_focus) == Chem.MolToSmiles(mol_seed), \
        'Input mol2 must be the same canonical mol'
    mol_seed_neworder = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol_seed))])))[1]
    mol_focus_neworder = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol_focus))])))[1]
    mol_focus_neworder = tuple(zip(*sorted(zip(mol_seed_neworder, mol_focus_neworder))))[1]
    mol_focus_renum = Chem.RenumberAtoms(mol_focus, mol_focus_neworder)
    mol_focus_renum_pos = mol_focus_renum.GetConformer().GetPositions()

    seed_conf = mol_seed.GetConformer()
    for at_id, at in enumerate(mol_seed.GetAtoms()):
        x, y, z = mol_focus_renum_pos[at_id]
        seed_conf.SetAtomPosition(at_id, Point3D(float(x), float(y), float(z)))

    return mol_seed