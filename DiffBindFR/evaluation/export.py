# Copyright (c) MDLDrugLib. All rights reserved.
import shutil, glob
from typing import (
    Optional, Union, Tuple,
    Mapping, Dict,
)
import os.path as osp
from pathlib import Path
from collections import defaultdict
from functools import partial

from rdkit import Chem
import MDAnalysis as mda

import pandas as pd
import numpy as np
import torch
from torch import Tensor

from druglib import mkdir_or_exists, digit_version
from druglib.utils.obj import Protein, Ligand3D, PLComplex
from DiffBindFR.metrics import (
    calc_lig_centroid,
    symm_rmsd,
    calc_rmsd,
    chi_differ,
    sidechain_rmsd,
)
from DiffBindFR.common import add_center_pos


def rmsd_to_str(rmsd):
    rmsd = round(rmsd, 2)
    rmsd = str(rmsd)
    rmsd = rmsd.replace('.', '_')
    return rmsd

def get_traj_id(file):
    idx = file.split('_')[-1].split('.')[0]
    return int(idx)

def pair(
        p: Protein,
        l: Ligand3D,
) -> PLComplex:
    return PLComplex(
        protein = p,
        ligand = l,
    )

def export_pdbstring(
        p: Union[PLComplex, Protein],
        output_p: str,
):
    """save the ligand-protein complex pdb file"""
    pdbstrings = p.to_pdb()
    with open(output_p, 'w') as f:
        f.write(pdbstrings)

    return pdbstrings

def update_complex_pos(
        PLComp,
        p_pos: Union[Tensor, np.ndarray],
        l_pos: Union[Tensor, np.ndarray],
        output_p: Optional[str] = None,
        enable_build_prot_model: bool = False,
        lig_hasRemovedHs: bool = True,
):
    NewComp = PLComp.pos_update(
        prot_new_pos = p_pos,
        lig_new_pos = l_pos,
        enable_build_prot_model = enable_build_prot_model,
        lig_hasRemovedHs = lig_hasRemovedHs,
    )

    if output_p is not None:
        output_dir = osp.dirname(osp.abspath(output_p))
        mkdir_or_exists(output_dir)
        export_pdbstring(NewComp, output_p)

    return NewComp

def export_xtc(
        topo_path: str,
        pdbs_dir: str,
        export_xtc: str,
):
    u = mda.Universe(
        topo_path,
        sorted(glob.glob(osp.join(pdbs_dir, '*.pdb')), key=get_traj_id),
    )
    ag = u.select_atoms("all")
    ag.write(export_xtc, frames='all')
    return

def mol2sdf(
        mol: Chem.rdchem.Mol,
        sdf: str,
):
    sdfwriter = Chem.SDWriter(sdf)
    sdfwriter.write(mol)
    sdfwriter.close()
    return

def complex_modeling(
        dataset: torch.utils.data.Dataset,
        export_dir: Optional[Union[str, Path]] = None,
        calc_metrics: bool = False,
        lrmsd_naming: bool = False,
        complex_name_split: Optional[str] = None,
        cp_raw: bool = False,
        **kwargs,
) -> Tuple[pd.DataFrame, Optional[Dict[str, np.ndarray]]]:
    pair_frame = dataset.pair_frame
    PD = dataset.PairData
    traj_group = PD.traj_group
    proteins = PD.proteins
    ligands = PD.ligands
    protein_name = dataset.protein_index
    ligand_name = dataset.ligand_index

    # decide whether export structures
    keywords = ['export_fullp', 'export_pkt', 'export_fullp_traj', 'export_pkt_traj']
    flags    = [kwargs.get(x, False) for x in keywords]

    PANDAS_VERSION = pd.__version__
    chi_upper_bound = 15 / 180 * torch.pi
    df, pd_df = defaultdict(list), defaultdict(list)
    for idx, (pl_name, one_traj) in enumerate(traj_group.items()):
        protein_key = pair_frame.iloc[idx][protein_name]
        ligand_key = pair_frame.iloc[idx][ligand_name]
        proteinmeta = proteins[protein_key]
        ligandmeta = ligands[ligand_key]

        ligand_traj = one_traj.ligand
        protein_traj = one_traj.protein
        pocket_center_pos = proteinmeta.pocket_center_pos
        ligand_traj = add_center_pos(ligand_traj, pocket_center_pos)
        protein_traj = add_center_pos(protein_traj, pocket_center_pos)
        assert ligand_traj.shape[0] == protein_traj.shape[0], \
            (f'The number poses of ligand ({ligand_traj.shape[0]}) '
             f'mismatch with protein ({protein_traj.shape[0]})')

        # copy input pair frame to output frame
        # iteritems was removed in 2.0.0 by https://github.com/pandas-dev/pandas/pull/45321.
        iters_series = pair_frame.iloc[idx].iteritems() \
            if digit_version(PANDAS_VERSION) < digit_version('2.0.0') \
            else pair_frame.iloc[idx].items()
        for k, v in iters_series:
            pd_df[k].extend([v] * ligand_traj.shape[0])

        if calc_metrics:
            # Evaluation on ligand rmsd (best, median), centroid, chi1_15 success rate, sc-RMSD (best, sc-median)
            # If only for analysis model performance, we focus on the last conformation over the sampled trajectories.

            # centroid
            centroid = calc_lig_centroid(
                ligand_traj, torch.from_numpy(ligandmeta.ligand.atom_positions).float())
            centroid_last = centroid[..., -1]
            df['centroid'].append(centroid_last.tolist())
            pd_df['centroid'].extend(centroid_last.tolist())

            # protein chi difference below 15 degree
            target_atom14_position = proteinmeta.atom14_position
            target_atom14_mask = proteinmeta.atom14_mask
            target_sequence = torch.LongTensor(proteinmeta.pocket.aatype)
            # (N_pose, N_traj, N_node, 4)
            delta_chi, sc_tor_mask = chi_differ(
                protein_traj,  # (N_pose, N_traj, N_node, 14, 3)
                target_atom14_position,
                target_atom14_mask,
                target_sequence,  # (N,)
            )
            # sc_success_rate
            criterion = (delta_chi < chi_upper_bound) * sc_tor_mask.bool()
            # reduce N_node to (N_pose, N_traj, 4); we focus on the chi1, so to (N_pose, N_traj)
            chi1_success_rate = (criterion.sum(dim=-2) / sc_tor_mask.sum(dim=-2))[..., 0]
            chi1_success_rate_last = chi1_success_rate[..., -1]
            df['chi1_15'].append(chi1_success_rate_last.tolist())
            pd_df['chi1_15'].extend(chi1_success_rate_last.tolist())

            # sc-RMSD
            sc_rmsd = sidechain_rmsd(
                protein_traj,
                add_center_pos(target_atom14_position, proteinmeta.pocket_center_pos),
                target_atom14_mask,
                target_sequence,
            )
            sc_rsmd_last = sc_rmsd[..., -1]
            df['sc-rmsd'].append(sc_rsmd_last.tolist())
            pd_df['sc-rmsd'].extend(sc_rsmd_last.tolist())

        # export structures
        if (not any(flags)) or export_dir is None: continue

        # export structural model for visualization
        protein = proteinmeta.protein
        pocket = proteinmeta.pocket
        pocket_mask = proteinmeta.pocket_mask
        ligand = ligandmeta.ligand

        PKLComp, PRLComp = pair(pocket, ligand), pair(protein, ligand)

        if complex_name_split is not None:
            # CrossDock complex names format as [Subset Name]:[CrossDock ID]
            pl_name = pl_name.split(complex_name_split)[-1]
        compl_dir = osp.join(export_dir, pl_name)
        mkdir_or_exists(compl_dir)

        # copy ground truth data to directory for structures comparison
        for x in (['ligand', 'protein'] if cp_raw else []):
            shutil.copy(pair_frame.iloc[idx][x], compl_dir)

        # export topology of xtc file
        if flags[-1]: export_pdbstring(PKLComp, osp.join(compl_dir, f'pkl_topol.pdb'))
        if flags[-2]: export_pdbstring(PRLComp, osp.join(compl_dir, f'prl_topol.pdb'))

        # export structure for every complex
        lrmsds = []
        for pid in range(len(ligand_traj)):
            sample_id = f'sample_{pid + 1}'
            # export final ligand pose
            ip_lig_traj = ligand_traj[pid]
            new_mol = ligand.pos_update(ip_lig_traj[-1], None, True)

            if calc_metrics:
                lrmsd = calc_rmsd(
                    new_mol.model,
                    ligandmeta.ligand.model,
                )
                # naming the file name with final ligand conformer rmsd
                if lrmsd_naming: sample_id += f'_{rmsd_to_str(lrmsd)}'
                lrmsds.append(lrmsd)
                pd_df['l-rmsd'].append(lrmsd)
            pd_df['sample_id'].append(sample_id)

            temp_data_path = osp.join(compl_dir, sample_id)
            mkdir_or_exists(temp_data_path)
            mol2sdf(
                new_mol.model,
                osp.join(temp_data_path, 'lig_final.sdf')
            )
            pd_df['docked_lig'].append(osp.join(temp_data_path, 'lig_final.sdf'))

            # export final protein structure
            ip_prot_traj = protein_traj[pid]
            fp14_pos, _ = protein.to_pos14(True)
            fp14_pos[pocket_mask] = ip_prot_traj[-1]
            if flags[0]:
                export_pdbstring(
                    protein.pos_update(fp14_pos, None, False),
                    osp.join(temp_data_path, 'prot_final.pdb')
                )

            if flags[1]:
                export_pdbstring(
                    pocket.pos_update(ip_prot_traj[-1], None, False),
                    osp.join(temp_data_path, 'pkt_final.pdb')
                )

            if flags[0]:
                pd_df['protein_pdb'].append(osp.join(temp_data_path, 'prot_final.pdb'))
            elif flags[1]:
                pd_df['protein_pdb'].append(osp.join(temp_data_path, 'pkt_final.pdb'))
            else:
                pd_df['protein_pdb'].append(pair_frame.iloc[idx]['protein'])

            if not any(flags[-2:]): continue

            # trajectory writer
            pkl_traj_temp_data_path = osp.join(temp_data_path, 'pkl_traj')
            prl_traj_temp_data_path = osp.join(temp_data_path, 'prl_traj')
            for tid in range(len(ip_lig_traj)):
                # export pocket-ligand structure
                if flags[-1]:
                    mkdir_or_exists(pkl_traj_temp_data_path)
                    update_complex_pos(
                        PKLComp,
                        ip_prot_traj[tid],
                        ip_lig_traj[tid],
                        output_p=osp.join(pkl_traj_temp_data_path, f'pkl_{tid}.pdb'),
                    )
                # export full protein-ligand structure
                if flags[-2]:
                    mkdir_or_exists(prl_traj_temp_data_path)
                    fp14_pos[pocket_mask] = ip_prot_traj[tid]
                    update_complex_pos(
                        PRLComp,
                        fp14_pos,
                        ip_lig_traj[tid],
                        output_p=osp.join(prl_traj_temp_data_path, f'prl_{tid}.pdb'),
                    )
            if flags[-1] and len(ip_lig_traj) > 0:
                export_xtc(
                    osp.join(compl_dir, 'pkl_topol.pdb'),
                    pkl_traj_temp_data_path,
                    osp.join(temp_data_path, 'pkl_traj.xtc'),
                )
            if flags[-2] and len(ip_lig_traj) > 0:
                export_xtc(
                    osp.join(compl_dir, 'prl_topol.pdb'),
                    prl_traj_temp_data_path,
                    osp.join(temp_data_path, 'prl_traj.xtc'),
                )

        df['l-rmsd'].append(lrmsds)

    pd_df = pd.DataFrame(pd_df)
    arr_df = {k: np.array(v) for k, v in df.items()} if calc_metrics else None
    return pd_df, arr_df

def _export_fn(
        row: pd.Series,
        export_dir: str,
        proteins: Mapping,
        ligands: Mapping,
        protein_name: str = 'protein_name',
        ligand_name: str = 'ligand_name',
        complex_index: str = 'complex_name',
        export_pkl: bool = True,
        override: bool = False,
):
    PID = row['PID']
    protein_key = row[protein_name]
    ligand_key = row[ligand_name]
    pl_name = row[complex_index]
    ligand_traj = row['ligand_traj']
    protein_traj = row['protein_traj']

    proteinmeta = proteins[protein_key]
    ligandmeta = ligands[ligand_key]

    pocket_center_pos = proteinmeta.pocket_center_pos
    ligand_traj = add_center_pos(ligand_traj, pocket_center_pos)
    protein_traj = add_center_pos(protein_traj, pocket_center_pos)

    # export structural model for visualization
    protein = proteinmeta.protein
    pocket_mask = proteinmeta.pocket_mask
    ligand = ligandmeta.ligand
    NPOS = len(ligand_traj)

    conf_count = NPOS * PID
    # export structure for every complex
    for pid in range(NPOS):
        key = f'{pl_name}-{str(pid).zfill(4)}_{conf_count}'
        if not override and all(
            [osp.exists(osp.join(export_dir, f'{key}.pdb')),
             osp.exists(osp.join(export_dir, f'{key}.sdf')),
             (osp.exists(osp.join(export_dir, f'{key}')) if export_pkl else True)]
        ):
            continue
        # select conformation
        ip_lig_traj = ligand_traj[pid]
        ip_prot_traj = protein_traj[pid]
        fp14_pos, _ = protein.to_pos14(True)
        fp14_pos[pocket_mask] = ip_prot_traj[-1]
        # export final protein structure
        export_pdbstring(
            protein.pos_update(fp14_pos, None, False),
            osp.join(export_dir, f'{key}.pdb')
        )
        new_mol = ligand.pos_update(ip_lig_traj[-1], None, True)
        mol2sdf(
            new_mol.model,
            osp.join(export_dir, f'{key}.sdf')
        )
        conf_count += 1

    return


def export_flatten_structures(
        dataset: torch.utils.data.Dataset,
        df: pd.DataFrame,
        export_dir: str,
        export_pkl: bool = True,
        num_workers: int = 12,
        verbose: bool = True,
        override: bool = False,
):
    PD = dataset.PairData
    process_fn = partial(
        _export_fn,
        export_dir = export_dir,
        proteins = PD.proteins,
        ligands = PD.ligands,
        protein_name = dataset.protein_index,
        ligand_name = dataset.ligand_index,
        complex_index = dataset.complex_index,
        export_pkl = export_pkl,
        override = override,
    )
    if num_workers > 1:
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=num_workers, progress_bar=verbose)
        apply_fn = df.parallel_apply
    else:
        from tqdm import tqdm
        tqdm.pandas()
        apply_fn = df.progress_apply if verbose else df.apply

    apply_fn(process_fn, axis = 1)

    return
