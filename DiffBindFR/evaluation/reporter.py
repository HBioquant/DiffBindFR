# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Optional, List, Dict, Union,
)
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

gold_cutoff = {
    'l-rmsd': 2.,
    'centroid': 1.,
    'chi1_15': 0.75,
    'sc-rmsd': 1.0,
}


def report_enrichment(
        eval_results_df: Dict[str, np.ndarray],
        top_ns: List[int] = [1, 3, 5, 10],
        decimals: int = 2,
        show_reports: bool = True,
):
    from prettytable import PrettyTable
    ptb = PrettyTable()

    interested_cutoff = {
        'l-rmsd': [1., 2., 2.5, 3, 4.],
        'centroid': [0.5, 1., 2.],
        'chi1_15': [0.80, 0.70, 0.60, 0.50],
        'sc-rmsd': [0.5, 1.0, 1.5, 2.0],
    }

    ptb.title = '\033[5;36mDiffBindFR Model evaluations\033[0m'
    ptb.field_names = ['\033[31mMetric\033[0m', '\033[34mPerformance\033[0m']
    ptb.align["\033[31mMetric\033[0m"] = "l"
    ptb.align["\033[34mPerformance\033[0m"] = "l"

    # all metrics are shape [N_complex, N_conf]
    l_rmsd = eval_results_df['l-rmsd']
    centroid = eval_results_df['centroid']
    chi1_15 = eval_results_df['chi1_15']
    sc_rmsd = eval_results_df['sc-rmsd']

    total_nposes = len(l_rmsd.flatten())
    ptb.add_row(['Total poses', f'{l_rmsd.shape[0]}x{l_rmsd.shape[1]}={total_nposes}'])

    # record gold standard
    for k, v in gold_cutoff.items():
        ptb.add_row([f'{k} gold standard', v])
    ptb.add_row(['', ''])

    # 1. report ligand-rmsd
    # represent enrichment
    mean = l_rmsd.mean().round(decimals)
    std = np.std(l_rmsd).round(decimals)
    median = np.median(l_rmsd).round(decimals)
    ptb.add_row(['All poses l-rmsd mean-std', f'{mean}-{std}'])
    ptb.add_row(['All poses l-rmsd median', median])
    for x in interested_cutoff['l-rmsd']:
        l_rmsd_belowx = (l_rmsd < x).sum()
        ptb.add_row([f'All poses l-rmsd count < {x}A', l_rmsd_belowx])
        ptb.add_row([f'All poses l-rmsd fraction < {x}A',
                     f'{np.round(l_rmsd_belowx / total_nposes * 100, decimals)}%'])

    # represent every complex
    success_rate = (l_rmsd < gold_cutoff["l-rmsd"]).astype(float).mean(axis = -1)
    mean = success_rate.mean().round(decimals)
    std = np.std(success_rate).round(decimals)
    median = np.median(success_rate).round(decimals)
    ptb.add_row([f'Each complex l-rmsd < {gold_cutoff["l-rmsd"]}A SR mean-std', f'{mean}-{std}'])
    ptb.add_row([f'Each complex l-rmsd < {gold_cutoff["l-rmsd"]}A SR median', median])

    for x in top_ns:
        permute = np.argsort(l_rmsd, axis = -1)
        top_n_metric = np.take_along_axis(l_rmsd, permute[..., :x], axis = -1)
        gold_fraction = np.all(
            top_n_metric < gold_cutoff["l-rmsd"], axis = -1
        ).astype(float).mean().round(decimals + 2)
        ptb.add_row(
            [f'top {x} l-rmsd all < {gold_cutoff["l-rmsd"]}A complex fraction',
             f'{gold_fraction * 100.}%'])

        # statistic of metric mapped to l-rmsd
        top_n_metric = np.take_along_axis(centroid, permute[..., :x], axis = -1)
        gold_fraction = np.all(
            top_n_metric < gold_cutoff["centroid"], axis=-1
        ).astype(float).mean().round(decimals + 2)
        ptb.add_row(
            [f'top {x} l-rmsd mapped centroid all < {gold_cutoff["centroid"]}A complex fraction',
             f'{gold_fraction * 100.}%'])

        top_n_metric = np.take_along_axis(chi1_15, permute[..., :x], axis = -1)
        gold_fraction = np.all(
            top_n_metric > gold_cutoff["chi1_15"], axis=-1
        ).astype(float).mean().round(decimals + 2)
        ptb.add_row(
            [f'top {x} l-rmsd mapped chi1 < 15 degree SR all < {gold_cutoff["chi1_15"] * 100}% complex fraction',
             f'{gold_fraction * 100.}%'])

        top_n_metric = np.take_along_axis(sc_rmsd, permute[..., :x], axis = -1)
        gold_fraction = np.all(
            top_n_metric < gold_cutoff["sc-rmsd"], axis=-1
        ).astype(float).mean().round(decimals + 2)
        ptb.add_row(
            [f'top {x} l-rmsd mapped sc-rmsd all < {gold_cutoff["sc-rmsd"]}A complex fraction',
             f'{gold_fraction * 100.}%'])
    ptb.add_row(['', ''])

    # 2. report ligand centroid
    # represent enrichment
    mean = centroid.mean().round(decimals)
    std = np.std(centroid).round(decimals)
    median = np.median(centroid).round(decimals)
    ptb.add_row(['All poses centroid mean-std', f'{mean}-{std}'])
    ptb.add_row(['All poses centroid median', median])
    for x in interested_cutoff['centroid']:
        centroid_belowx = (centroid < x).sum()
        ptb.add_row([f'All poses centroid distance count < {x}A', centroid_belowx])
        ptb.add_row([f'All poses centroid fraction < {x}A',
                     f'{np.round(centroid_belowx / total_nposes * 100, decimals)}%'])

    # represent every complex
    success_rate = (centroid < gold_cutoff["centroid"]).astype(float).mean(axis=-1)
    mean = success_rate.mean().round(decimals)
    std = np.std(success_rate).round(decimals)
    median = np.median(success_rate).round(decimals)
    ptb.add_row([f'Each complex centroid < {gold_cutoff["centroid"]}A SR mean-std', f'{mean}-{std}'])
    ptb.add_row([f'Each complex centroid < {gold_cutoff["centroid"]}A SR median', median])

    for x in top_ns:
        permute = np.argsort(centroid, axis=-1)
        top_n_metric = np.take_along_axis(centroid, permute[..., :x], axis = -1)
        gold_fraction = np.all(
            top_n_metric < gold_cutoff["centroid"], axis=-1
        ).astype(float).mean().round(decimals + 2)
        ptb.add_row(
            [f'top {x} centroid all < {gold_cutoff["centroid"]} angstrom complex fraction',
             f'{gold_fraction * 100.}%'])
    ptb.add_row(['', ''])

    # 3. report chi1 diference below 15 degree success rate
    # represent enrichment
    mean = chi1_15.mean().round(decimals)
    std = np.std(chi1_15).round(decimals)
    median = np.median(chi1_15).round(decimals)
    ptb.add_row(['All conf chi1 15 SR mean-std', f'{mean}-{std}'])
    ptb.add_row(['All conf chi1 15 SR median', median])
    for x in interested_cutoff['chi1_15']:
        chi1_15_abovex = (chi1_15 > x).sum()
        ptb.add_row([f'All conf chi1 15 SR count > {x * 100}%', chi1_15_abovex])
        ptb.add_row([f'All conf chi1 15 SR fraction > {x * 100}%',
                     f'{np.round(chi1_15_abovex / total_nposes * 100, decimals)}%'])

    # represent every complex
    success_rate = (chi1_15 > gold_cutoff["chi1_15"]).astype(float).mean(axis=-1)
    mean = success_rate.mean().round(decimals)
    std = np.std(success_rate).round(decimals)
    median = np.median(success_rate).round(decimals)
    ptb.add_row(
        [f'Each complex chi1 15 SR > {gold_cutoff["chi1_15"] * 100}% SR mean-std', f'{mean}-{std}'])
    ptb.add_row([f'Each complex chi1 15 SR > {gold_cutoff["chi1_15"] * 100}% SR median', median])

    for x in top_ns:
        permute = np.argsort(chi1_15, axis=-1)
        top_n_metric = np.take_along_axis(chi1_15, permute[..., -x:], axis = -1)
        gold_fraction = np.all(
            top_n_metric > gold_cutoff["chi1_15"], axis=-1
        ).astype(float).mean().round(decimals + 2)
        ptb.add_row(
            [f'top {x} chi1 15 SR all > {gold_cutoff["chi1_15"] * 100}% complex fraction',
             f'{gold_fraction * 100.}%'])
    ptb.add_row(['', ''])

    # 4. report side chain rmsd
    # represent enrichment
    mean = sc_rmsd.mean().round(decimals)
    std = np.std(sc_rmsd).round(decimals)
    median = np.median(sc_rmsd).round(decimals)
    ptb.add_row(['All conf sc-rmsd mean-std', f'{mean}-{std}'])
    ptb.add_row(['All conf sc-rmsd median', median])
    for x in interested_cutoff['sc-rmsd']:
        sc_rmsd_belowx = (sc_rmsd < x).sum()
        ptb.add_row([f'All conf sc-rmsd count < {x}A', sc_rmsd_belowx])
        ptb.add_row([f'All conf sc-rmsd fraction < {x}A',
                     f'{np.round(sc_rmsd_belowx / total_nposes * 100, decimals)}%'])

    # represent every complex
    success_rate = (sc_rmsd < gold_cutoff["sc-rmsd"]).astype(float).mean(axis=-1)
    mean = success_rate.mean().round(decimals)
    std = np.std(success_rate).round(decimals)
    median = np.median(success_rate).round(decimals)
    ptb.add_row(
        [f'Each complex sc-rmsd < {gold_cutoff["sc-rmsd"]}A SR mean-std', f'{mean}-{std}'])
    ptb.add_row([f'Each complex sc-rmsd < {gold_cutoff["sc-rmsd"]}A SR median', median])

    for x in top_ns:
        permute = np.argsort(sc_rmsd, axis=-1)
        top_n_metric = np.take_along_axis(sc_rmsd, permute[..., :x], axis = -1)
        gold_fraction = np.all(
            top_n_metric < gold_cutoff["sc-rmsd"], axis=-1
        ).astype(float).mean().round(decimals + 2)
        ptb.add_row(
            [f'top {x} sc-rmsd all < {gold_cutoff["sc-rmsd"]}A complex fraction',
             f'{gold_fraction * 100.}%'])

    if show_reports:
        print(ptb, flush = True)

    return ptb

def report_performance(
        summary_df: Union[pd.DataFrame, str, Path],
        ec_flag: Optional[bool] = None,
):
    if isinstance(summary_df, (str, Path)):
        summary_df = pd.read_csv(summary_df)
    pocket_index = 'center' if 'center' in summary_df.columns else 'crystal_ligand'

    best_dfs = summary_df.loc[
        summary_df.groupby(
            ['complex_name', 'ligand', pocket_index], sort=False,
        )['l-rmsd'].agg('idxmin')
    ].reset_index(drop=True)
    N = best_dfs['l-rmsd'].shape[0]
    print('Total number of DiffBindFR perfect poses:', N)
    print('The DiffBindFR perfect selected L-RMSD success rate:',
          round((best_dfs['l-rmsd'] <= gold_cutoff['l-rmsd']).sum() / N * 100, 1), '%')
    print('The DiffBindFR perfect selected sc-RMSD success rate:',
          round((best_dfs['sc-rmsd'] <= gold_cutoff['sc-rmsd']).sum() / N * 100, 1), '%')
    print('The DiffBindFR perfect selected centroid success rate:',
          round((best_dfs['centroid'] <= gold_cutoff['centroid']).sum() / N * 100, 1), '%')
    print('The DiffBindFR perfect selected chi1_15 success rate:',
          round((best_dfs['chi1_15'] >= gold_cutoff['chi1_15']).sum() / N * 100, 1), '%')
    print()

    if ec_flag is None:
        ec_flag = True if 'l-rmsd_ec' in summary_df.columns else False

    if ec_flag:
        print('Use error corrected poses...')
        rmsd_nm = 'l-rmsd_ec'
        cdist_nm = 'centroid_ec'
        best_dfs = summary_df.loc[
            summary_df.groupby(
                ['complex_name', 'ligand', pocket_index], sort=False
            )['l-rmsd_ec'].agg('idxmin')
        ].reset_index(drop=True)
        N = best_dfs['l-rmsd_ec'].shape[0]
        print('Total number of DiffBindFR-Smina perfect poses:', N)
        print('The DiffBindFR-Smina perfect selected success rate:',
              round((best_dfs['l-rmsd_ec'] <= gold_cutoff['l-rmsd']).sum() / N * 100, 1), '%')
        print('The DiffBindFR-Smina perfect selected sc-rmsd success rate:',
              round((best_dfs['sc-rmsd'] <= gold_cutoff['sc-rmsd']).sum() / N * 100, 1), '%')
        print('The DiffBindFR-Smina perfect selected centroid success rate:',
              round((best_dfs['centroid_ec'] <= gold_cutoff['centroid']).sum() / N * 100, 1), '%')
        print('The DiffBindFR-Smina perfect selected chi1_15 success rate:',
              round((best_dfs['chi1_15'] >= gold_cutoff['chi1_15']).sum() / N * 100, 1), '%')
        print()
        best_dfs = summary_df.loc[
            summary_df.groupby(
                ['complex_name', 'ligand', pocket_index], sort=False
            )['smina_score'].agg('idxmin')
        ].reset_index(drop=True)
        N = best_dfs['l-rmsd_ec'].shape[0]
        print('Total number of DiffBindFR-Smina top1 poses:', N)
        print('The DiffBindFR-Smina top1 selected success rate:',
              round((best_dfs['l-rmsd_ec'] <= gold_cutoff['l-rmsd']).sum() / N * 100, 1), '%')
        print('The DiffBindFR-Smina top1 selected sc-rmsd success rate:',
              round((best_dfs['sc-rmsd'] <= gold_cutoff['sc-rmsd']).sum() / N * 100, 1), '%')
        print('The DiffBindFR-Smina top1 selected centroid success rate:',
              round((best_dfs['centroid_ec'] <= gold_cutoff['centroid']).sum() / N * 100, 1), '%')
        print('The DiffBindFR-Smina top1 selected chi1_15 success rate:',
              round((best_dfs['chi1_15'] >= gold_cutoff['chi1_15']).sum() / N * 100, 1), '%')
        print()
    else:
        print('Use model sampled poses...')
        rmsd_nm = 'l-rmsd'
        cdist_nm = 'centroid'

    if 'mdn_score' in summary_df.columns:
        best_dfs = summary_df.loc[
            summary_df.groupby(
                ['complex_name', 'ligand', pocket_index], sort=False
            )['mdn_score'].agg('idxmax')
        ].reset_index(drop=True)
        N = best_dfs[rmsd_nm].shape[0]
        print('Total number of DiffBindFR-MDN top1 poses:', N)
        print('The DiffBindFR-MDN top1 l-rmsd success rate:',
              round((best_dfs[rmsd_nm] <= gold_cutoff['l-rmsd']).sum() / N * 100, 1), '%')
        print('The DiffBindFR-MDN top1 sc-rmsd success rate:',
              round((best_dfs['sc-rmsd'] <= gold_cutoff['sc-rmsd']).sum() / N * 100, 1), '%')
        print('The DiffBindFR-MDN top1 centroid success rate:',
              round((best_dfs[cdist_nm] <= gold_cutoff['centroid']).sum() / N * 100, 1), '%')
        print('The DiffBindFR-MDN top1 chi1_15 success rate:',
              round((best_dfs['chi1_15'] >= gold_cutoff['chi1_15']).sum() / N * 100, 1), '%')

    print('Report perfect and top 1 performance is done!')
    return

### PoseBusters report
pb_metrics = [
    'rmsd_≤_2å',
    'sanitization', 'all_atoms_connected', 'molecular_formula', 'molecular_bonds',
    'bond_angles', 'aromatic_ring_flatness', 'double_bond_flatness', 'protein-ligand_maximum_distance',
    'double_bond_stereochemistry','tetrahedral_chirality', 'internal_steric_clash',
    'internal_energy', 'bond_lengths',
    'volume_overlap_with_inorganic_cofactors',
    'volume_overlap_with_organic_cofactors',
    'minimum_distance_to_inorganic_cofactors',
    'minimum_distance_to_organic_cofactors',
    'minimum_distance_to_waters',
    'volume_overlap_with_waters',
    'volume_overlap_with_protein',
    'minimum_distance_to_protein',
]

def report_pb(
        best_df: pd.DataFrame,
        expected_pose_number: Optional[int] = None,
):
    total_poses = best_df.shape[0]
    if expected_pose_number is None:
        expected_pose_number = total_poses

    print('total poses number: ', total_poses)
    print('Expected poses number: ', expected_pose_number)
    if (expected_pose_number - total_poses) != 0:
        print('!!!!!!!Failed poses number: ', expected_pose_number - total_poses)
    print('rmsd mean: ', np.mean(best_df['l-rmsd']))
    print('rmsd nan mean: ', np.nanmean(best_df['l-rmsd']))
    print('rmsd std: ', np.std(best_df['l-rmsd']))
    print('rmsd nan std: ', np.nanstd(best_df['l-rmsd']))
    print('rmsd median: ', np.nanmedian(best_df['l-rmsd']))
    # print(np.nanmean(best_df['l-rmsd']))
    # print(np.nanstd(best_df['l-rmsd']))
    # print(np.nanmedian(best_df['l-rmsd']))
    before_performance = [1.]
    # print_df = defaultdict(list)
    print_detail = defaultdict(list)
    for i in range(len(pb_metrics)):
        results = best_df.copy()[pb_metrics[:i + 1]]
        results.columns = results.columns.to_flat_index()
        columns = results.columns
        pass_df = results[columns].all(axis=1)
        pass_num = pass_df.sum()
        perf = pass_num / expected_pose_number
        # print('to ', pb_metrics[i], pass_num, round(perf, 3), f'({round(perf - before_performance[i], 3)})')
        before_performance.append(perf)
        # print_df['num'].append(pass_num)

        print_detail['metric'].append(pb_metrics[i])
        print_detail['num'].append(pass_num)
        print_detail['sr'].append(round(perf, 3))
        print_detail['error'].append(round(perf - before_performance[i], 3))

    # print(pd.DataFrame(print_df).to_string(index=False))
    print(pd.DataFrame(print_detail).to_string(index=False))

    return