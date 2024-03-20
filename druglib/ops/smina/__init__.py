# Copyright (c) MDLDrugLib. All rights reserved.
import os
from typing import Optional, List
import os.path as osp
import subprocess
from pathlib import Path

import numpy as np
from rdkit import Chem

# User defined Bin path
this_file = osp.abspath(__file__)
this_dir = osp.dirname(this_file)
Smina_bin = osp.join(this_dir, 'smina.static')

def get_smina_score(file) -> float:
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(file)
    mol = Chem.SDMolSupplier(str(file), sanitize = False)[0]
    score = float(mol.GetProp('minimizedAffinity'))
    return score

def smina_min(
        receptor_path: str,
        ligand_path: str,
        out_path: str,
        smina_bin: str = Smina_bin,
        score_only: bool = False,
        exhaustiveness: Optional[int] = None,
) -> float:
    for x in [receptor_path, ligand_path]:
        if not Path(x).exists():
            raise FileNotFoundError(x)

    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    option = 'score_only' if score_only else 'minimize'
    if exhaustiveness is not None and isinstance(exhaustiveness, int):
        option = f'exhaustiveness {exhaustiveness}'

    if out_path.exists() and osp.getsize(str(out_path)) == 0:
        os.remove(str(out_path))

    if not out_path.exists():
        subprocess.check_call(
            f'{smina_bin} '
            f'-r {receptor_path} '
            f'-l {ligand_path} '
            f'--autobox_ligand {ligand_path} '
            f'--{option} '
            f'-o {str(out_path)} '
            f'--quiet '
            f'>/dev/null 2>&1',
            shell=True,
            stdout=subprocess.DEVNULL,
        )
    if not out_path.exists():
        print(f'No smina minimized pose is output for {out_path}. Return inf.')
        return np.inf

    score = get_smina_score(out_path)
    return score

def smina_min_forward(
        protein_file,
        smina_bin: str = Smina_bin,
        score_only: bool = False,
        exhaustiveness: Optional[int] = None,
) -> float:
    if not Path(protein_file).exists():
        raise FileNotFoundError(protein_file)
    option = 'score_only' if score_only else 'minimize'
    if exhaustiveness is not None and isinstance(exhaustiveness, int):
        option = f'exhaustiveness {exhaustiveness}'
    work_dir = Path(protein_file).parent
    profile_name = Path(protein_file).name
    ligfile_name = f'{Path(protein_file).stem}.sdf'
    outfile_name = f'{Path(protein_file).stem}_min.sdf'
    op = osp.join(str(work_dir), outfile_name)
    if osp.exists(op) and osp.getsize(op) == 0:
        os.remove(op)

    if not osp.exists(op):
        subprocess.check_call(
            f'cd {str(work_dir)} && '
            f'{smina_bin} '
            f'-r {profile_name} '
            f'-l {ligfile_name} '
            f'--autobox_ligand {ligfile_name} '
            f'--{option} '
            f'-o {outfile_name} '
            f'--quiet '
            f'>/dev/null 2>&1',
            shell=True,
            stdout=subprocess.DEVNULL,
        )

    if not osp.exists(op):
        print(f'No smina minimized pose is output for {outfile_name}.')
        return np.inf

    score = get_smina_score(op)
    return score

def smina_min_inplace(
        work_dir,
        rec_name: str = 'protein.pdb',
        lig_name: str = 'ligand.sdf',
        out_name: str = 'output.sdf',
        smina_bin: str = Smina_bin,
        score_only: bool = False,
        exhaustiveness: Optional[int] = None,
):
    option = 'score_only' if score_only else 'minimize'
    if exhaustiveness is not None and isinstance(exhaustiveness, int):
        option = f'exhaustiveness {exhaustiveness}'

    work_dir = Path(work_dir)
    for x in [rec_name, lig_name]:
        if not (work_dir / x).exists():
            raise FileNotFoundError(work_dir / x)
    out_path = work_dir / out_name
    if out_path.exists() and osp.getsize(str(out_path)) == 0:
        os.remove(str(out_path))

    if not out_path.exists():
        subprocess.check_call(
            f'cd {work_dir} && '
            f'{smina_bin} '
            f'-r {rec_name} '
            f'-l {lig_name} '
            f'--autobox_ligand {lig_name} '
            f'--{option} '
            f'-o {out_name} '
            f'--quiet '
            f'>/dev/null 2>&1',
            shell=True,
            stdout=subprocess.DEVNULL,
        )
    if not out_path.exists():
        print(f'No smina minimized pose is output for {out_path}. Return inf.')
        return np.inf

    score = get_smina_score(out_path)
    return score
