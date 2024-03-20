# Copyright (c) MDLDrugLib. All rights reserved.
import os.path as osp
from rdkit import Chem
from openbabel import pybel
from ..path import mkdir_or_exists


def pdbqt2pdbblock(
        pdbqt_file: str,
) -> str:
    """
    Extracts the PDB part of a pdbqt file to pdb block.
    Strip pdbqt charge information from the provided input.
    Args:
        pdbqt_file: str. Filename of pdbqt file.
    Returns: pdb  block, str.
    """
    assert osp.exists(pdbqt_file), f"PDBQT File does not exists from {pdbqt_file}"

    with open(pdbqt_file, 'r') as f:
        pdbqt = f.readlines()

    pdb_block = ''
    for line in pdbqt:
        pdb_block += f'{line[:66]}\n'

    return pdb_block

def pdb2pdbqt(
        pdb_file: str,
        out_file: str,
) -> str:
    """
    Convert pdb file to pdbqt file.
    Write extra the pdbqt terms into the pdb file.
    Args:
        pdb_file: str. pdb file path.
        out_file: str. output pdbqt file path.
    Returns:
        out_file, str.
    **Note that we suggest using vina prepare_receptor and
        prepare_ligand to transform pdb file to pdbqt file.
    """
    assert osp.exists(pdb_file), f"PDB File does not exists from {pdb_file}"
    mkdir_or_exists(osp.dirname(osp.abspath(out_file)))

    mol = Chem.MolFromPDBFile(
        pdb_file,
        sanitize = True,
        removeHs = False,
    )
    lines = [line.strip() for line in open(pdb_file).readlines()]
    pdbqt_lines = []
    for line in lines:
        if 'ROOT' in line or 'ENDROOT' in line or 'TORSDOF' in line:
            pdbqt_lines.append(f'{line}\n')
            continue
        if not line.startswith("ATOM"):
            continue
        line = line[:66]
        atom_index = int(line[6:11])
        atom = mol.GetAtoms()[atom_index - 1]
        line = "%s    +0.000 %s\n" % (line, atom.GetSymbol().ljust(2))
        pdbqt_lines.append(line)
    with open(out_file, 'w') as f:
        for line in pdbqt_lines:
            f.write(line)

    return out_file

def pdbqt2sdf(
        pdbqt_file: str,
        out_file: str,
        log_level: int = 0,
):
    """
    A simple implementation about format transformation from pdbqt to sdf file.
    Args:
        pdbqt_file: str. pdbqt file path.
        out_file: str. output sdf file path.
        log_level: int. Log level, set 0 to slience warning.
    Returns:
        out_file, str.
    """
    assert osp.exists(pdbqt_file), f"PDBQT File does not exists from {pdbqt_file}"
    mkdir_or_exists(osp.dirname(osp.abspath(out_file)))

    pybel.ob.obErrorLog.SetOutputLevel(log_level)

    results = [m for m in pybel.readfile(format = 'pdbqt', filename = pdbqt_file)]
    outfile = pybel.Outputfile(
        filename = out_file,
        format = 'sdf',
        overwrite = True
    )
    for pose in results:
        pose.data.update(
            {
                'Pose' : pose.data['MODEL'],
                'Score':pose.data['REMARK'].split()[2]
             })
        del pose.data['MODEL'], pose.data['REMARK'], pose.data['TORSDO']

        outfile.write(pose)

    outfile.close()

    return out_file