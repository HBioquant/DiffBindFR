# Copyright (c) MDLDrugLib. All rights reserved.
import os
from typing import Union
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from rdkit import Chem
from prody import parsePDB, writePDB

def extract_contact_chains(
        protein_file: str,
        ligand_file: str,
        pocket_align_cutoff: float = 10.0,
):
    mol = Chem.SDMolSupplier(ligand_file)[0] # read_mol(ligand_file)
    mol = Chem.RemoveAllHs(mol)
    ligpos = mol.GetConformer().GetPositions()
    protein_prody_obj = parsePDB(protein_file)
    condition = f'same residue as exwithin {pocket_align_cutoff} of ligpoint'
    selected = protein_prody_obj.select(
        condition,
        ligpoint=ligpos,
    )
    unique_chains = np.unique(selected.toAtomGroup().getChids())
    selected = protein_prody_obj.select(f'chain ' + ' '.join(map(str, unique_chains.tolist())))

    return selected

def make_jobs_tstest(
        data_root: Union[str, Path],
        test_file_name: str = 'timesplit_test',
) -> pd.DataFrame:
    data_root = Path(data_root)
    testset = data_root / test_file_name
    if not testset.exists():
        raise FileNotFoundError(testset)

    with open(testset, "r") as f:
        val_lines = f.readlines()
        val_lines = [line.strip() for line in val_lines]

    pd_data = defaultdict(list)
    for pdbid in val_lines:
        ligand_file = str(data_root / pdbid / f'{pdbid}_ligand.sdf')
        protein_file = str(data_root / pdbid / f'{pdbid}_fix.pdb')
        pd_data['protein'].append(protein_file)
        pd_data['protein_name'].append(pdbid)
        # it does not matter as we will randomly initialize the mol conformer
        # generated from rdkit or changing conformer torsion.
        pd_data['ligand'].append(ligand_file)
        pd_data['ligand_name'].append(pdbid)
        pd_data['complex_name'].append(pdbid)
        pd_data['crystal_ligand'].append(ligand_file)

    pd_data = pd.DataFrame(pd_data)
    return pd_data

def make_jobs_pbtest(
        data_root: Union[str, Path],
) -> pd.DataFrame:
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            'Please make sure PoseBusters benchmark set has been downloaded and correctly specified.'
        )

    pd_data = defaultdict(list)
    for pb_id in sorted(os.listdir(data_root)):
        ligand_file = str(data_root / pb_id / f'{pb_id}_ligand.sdf')
        protein_file = str(data_root / pb_id / f'{pb_id}_protein.pdb')
        contact_chains_file = data_root / pb_id / f'{pb_id}_protein_contact_chains.pdb'
        if not contact_chains_file.exists():
            contact_chains = extract_contact_chains(
                protein_file, ligand_file, 10.0,
            )
            writePDB(str(contact_chains_file), contact_chains)

        # as pdb file has limit in chain number (exceed "Z" will raise error in our program)
        # TODO: fix the chain number limit bug in the future.
        # pd_data['protein'].append(protein_file)
        pd_data['protein'].append(str(contact_chains_file))
        pd_data['protein_name'].append(pb_id)
        # it does not matter as we will randomly initialize the mol conformer
        # generated from rdkit or changing conformer torsion.
        pd_data['ligand'].append(ligand_file)
        pd_data['ligand_name'].append(pb_id)
        pd_data['complex_name'].append(pb_id)
        pd_data['crystal_ligand'].append(ligand_file)

    pd_data = pd.DataFrame(pd_data)
    return pd_data

def make_jobs_cdtest(
        data_root: Union[str, Path],
        lib: str,
) -> pd.DataFrame:
    data_root = Path(data_root) / lib
    if not data_root.exists():
        raise FileNotFoundError(
            f'crossdock subset {lib} does not exists at {data_root}.'
        )

    pd_data = defaultdict(list)
    for cid in sorted(os.listdir(data_root)):
        ligand_file = data_root / cid / f'ligand.sdf'
        protein_file = data_root / cid / f'protein.pdb'
        pd_data['protein'].append(str(protein_file))
        pd_data['protein_name'].append(cid)
        # it does not matter as we will randomly initialize the mol conformer
        # generated from rdkit or changing conformer torsion.
        pd_data['ligand'].append(str(ligand_file))
        pd_data['ligand_name'].append(cid)
        pd_data['complex_name'].append(cid)
        pd_data['crystal_ligand'].append(str(ligand_file))

    pd_data = pd.DataFrame(pd_data)
    return pd_data

if __name__ == '__main__':
    # if you find there are errors raised when DiffBindFR evaluations
    # probably the number of protein chain exceeds limits
    # you can extract the contact chains use this scripts
    # you'd better use func:`read_mol` to extract the contact chain
    # as most of PDBBind ligand files are rdkit-unreadable
    from DiffBindFR.utils import read_mol
    # do something...