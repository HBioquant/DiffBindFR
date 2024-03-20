# Copyright (c) MDLDrugLib. All rights reserved.
import os, sys, shutil, argparse, logging
from typing import Optional, Union, List, Tuple, Dict, Any
import os.path as osp
from pathlib import Path
from collections import OrderedDict
from pandarallel import pandarallel

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from druglib.utils import DictAction, get_logger
from DiffBindFR.scoring import InferenceScoringDataset_chunk
from DiffBindFR.metrics import (
    caltestset_cdist, caltestset_rmsd,
)
from DiffBindFR import common
from DiffBindFR.evaluation import (
    complex_modeling,
    report_enrichment,
    report_performance,
)

logger = get_logger(name = 'Evaluator')


def out_fn(
        dl: DataLoader,
        pairs_results: List[Tuple[Any, Any]],
        results_out: Optional[Union[Path, str]] = None,
        struct_out: Optional[Union[Path, str]] = None,
        export_pocket: bool = False,
        override: bool = False,
        vis_traj: bool = False,
        calc_metrics: bool = False,
        show_reports: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    if not override and (results_out is not None and all((Path(results_out) / x).exists()
            for x in ['results_df.pt', 'results.csv', 'metrics_report.csv'])):
        from prettytable import from_csv
        logger.info(f'Reload evaluation results from {results_out}')
        eval_results_df = torch.load(str(Path(results_out) / 'results_df.pt'))
        pd_df = pd.read_csv(str(Path(results_out) / 'results.csv'))

        if show_reports:
            with open(str(Path(results_out) / 'metrics_report.csv'), 'r') as fp:
                ptb = from_csv(
                    fp, align = 'l',
                    title = '\033[5;36mDiffBindFR Model evaluations\033[0m',
                )
                print(ptb, flush = True)

        return pd_df, eval_results_df

    dataset = dl.dataset
    dataset.PairData.traj_group = OrderedDict()
    for idx, (names, rses) in enumerate(pairs_results):
        assert len(set(names)) == 1, f'{names} has more than one complex to match'
        name = names[0]
        ligand_traj, protein_traj = list(zip(*rses))
        ligand_traj = torch.stack(ligand_traj, dim = 0) # (N_pose, N_traj, N_latom, 3)
        protein_traj = torch.stack(protein_traj, dim = 0) # (N_pose, N_traj, N_patom, 3)
        dataset.PairData.put_inference(
            name,
            ligand_traj,
            protein_traj,
        )

    logger.info('Export binding structures....')
    pd_df, eval_results_df = complex_modeling(
        dataset = dataset,
        export_dir = str(struct_out) if struct_out is not None else None,
        calc_metrics = calc_metrics,
        lrmsd_naming = False,
        complex_name_split=':',
        export_fullp = True,
        export_pkt = export_pocket,
        export_fullp_traj = vis_traj,
        export_pkt_traj = (export_pocket and vis_traj),
    )
    logger.info('Binding structure export is completed.')
    if calc_metrics:
        logger.info('Start to binding conformation enrichment analysis...')
        ptb = report_enrichment(eval_results_df, show_reports = show_reports)

    if results_out is not None:
        results_out.mkdir(parents = True, exist_ok = True)
        pd_df.to_csv(str(Path(results_out) / 'results.csv'), index = False)
        if calc_metrics:
            torch.save(eval_results_df, str(Path(results_out) / 'results_df.pt'))
            (Path(results_out) / 'metrics_report.csv').write_text(ptb.get_csv_string(delimiter='\t'))

    return pd_df, eval_results_df

def runner(
        df: pd.DataFrame,
        args: argparse.Namespace,
):
    if args.debug: logger.setLevel(logging.DEBUG)

    cfg = common.load_cfg(args, logger)
    batch_size = cfg.batch_size if args.batch_size is None else args.batch_size
    dl = common.load_dataloader(
        df, args, cfg, logger,
        batch_size = batch_size,
        batch_repeat = False,
        lmdb_mode = True,
        debug = args.debug,
    )

    logger.info(f'{args.job} Status: Prep task is Done!')
    if args.job != 'dock':
        sys.exit(0)

    model = common.load_model(
        args, cfg,
        logger = logger,
        strict = True,
        map_location = 'cpu',
        drop_keys = [r'^ema_'],
        use_ema = False,
    )

    export_dir = args.export_dir / args.experiment_name
    results_dir = export_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    model_out_path = results_dir / 'model_output.pt'
    pairs_results = common.inferencer(
        dl, model,
        args.show_traj,
        args.override,
        model_out_path,
        logger,
    )

    calc_metrics = args.evaluation
    show_reports = args.report_performance
    if not calc_metrics:
        show_reports = False
    struct_dir = export_dir / 'structures'
    struct_dir.mkdir(parents=True, exist_ok=True)
    pd_df, eval_results_df = out_fn(
        dl = dl,
        pairs_results = pairs_results,
        results_out = results_dir,
        struct_out = struct_dir,
        export_pocket = args.export_pocket,
        override = args.override,
        vis_traj = args.show_traj,
        calc_metrics = calc_metrics,
        show_reports = show_reports,
    )

    pocket_index = dl.dataset.pocket_index
    pandarallel.initialize(nb_workers=args.num_workers, progress_bar=args.verbose)
    output_csv_path = results_dir / 'results.csv' if args.no_error_correction else results_dir / f'results{common.ec_tag}.csv'
    if not args.no_error_correction:
        logger.info('Start to correct error...')
        pd_df['smina_score'] = pd_df['docked_lig'].parallel_apply(
            lambda x: common.error_corrector(x, override = args.override)
        )
        pd_df['docked_lig'] = pd_df['docked_lig'].apply(
            lambda x: osp.join(
                Path(x).parent, Path(x).stem + common.ec_tag + Path(x).suffix
            )
        )
        logger.info('Error correction is completed.')
        if calc_metrics:
            pd_df['l-rmsd' + common.ec_tag] = pd_df.parallel_apply(
                lambda row: caltestset_rmsd(row['docked_lig'], row['ligand']),
                axis = 1
            )
            pd_df['centroid' + common.ec_tag] = pd_df.parallel_apply(
                lambda row: caltestset_cdist(row['docked_lig'], row['ligand']),
                axis = 1,
            )
        pd_df.to_csv(output_csv_path, index=False)
        top1_pd_df = pd_df.loc[
            pd_df.groupby(
                ['complex_name', 'ligand', pocket_index], sort=False
            )['smina_score'].agg('idxmin')
        ].reset_index(drop=True)
        top1_pd_df.to_csv(
            osp.join(
                Path(output_csv_path).parent, Path(output_csv_path).stem + '_smina_top1' + Path(output_csv_path).suffix,
            ),
            index = False,
        )

    if not args.no_mdn_scoring:
        logger.info('Start to MDN based pose scoring...')
        lmdb_save_path = export_dir / 'data' / 'mdn.lmdb' if args.no_error_correction else export_dir / 'data' / f'mdn{common.ec_tag}.lmdb'
        test_dataset = InferenceScoringDataset_chunk(
            pair_frame=pd_df,
            debug=True,
            pocket_radius=args.mdn_pocket_radius,
            save_path=str(lmdb_save_path),
            lmdb_mode=True,
            n_jobs=args.num_workers,
            verbose=args.verbose,
            chunk=50000,
            suffix='' if args.no_error_correction else 'Min',
        )
        common.Scorer(
            test_dataset,
            output_path=output_csv_path,
            batch_size=args.batch_size,
            device_id=args.gpu_id,
            logger=logger,
        )
        pd_df = pd.read_csv(output_csv_path)
        top1_pd_df = pd_df.loc[
            pd_df.groupby(
                ['complex_name', 'ligand', pocket_index], sort=False
            )['mdn_score'].agg('idxmax')
        ].reset_index(drop=True)
        top1_pd_df.to_csv(
            osp.join(
                Path(output_csv_path).parent, Path(output_csv_path).stem + '_mdn_top1' + Path(output_csv_path).suffix,
            ),
            index=False,
        )
        if show_reports: report_performance(output_csv_path)
        logger.info('MDN based pose scoring is completed.')

    if args.cleanup:
        logger.info('Clean up datasets and model_out.pt.')
        if (export_dir / 'data').exists():
            shutil.rmtree(export_dir / 'data')
        if model_out_path.exists():
            os.remove(model_out_path)
    logger.info('DiffBindFR docking is Done!')
    return

if __name__ == '__main__':
    parser = common.benchmark_parse_args()
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    args.evaluation = True
    args.report_performance = True
    if args.debug: common.report_args(args)

    def make_jobs(args: argparse.Namespace) -> pd.DataFrame:
        import DiffBindFR.evaluation.file_utils as fu
        logger.info(f'Use benchmark libs: {args.lib}')

        if args.lib in ['pdbbind_ts']:
            # PDBbindv2020 time split test set downloaded from https://zenodo.org/records/6408497
            pair_frame = fu.make_jobs_tstest(
                args.data_dir, 'timesplit_test',
            )
        elif args.lib in ['pb']:
            # PoseBusters benchmark set downloaded from https://zenodo.org/records/8278563
            pair_frame = fu.make_jobs_pbtest(args.data_dir)
        else:
            # CrossDock benchmark set downloaded from ...
            pair_frame = fu.make_jobs_cdtest(
                args.data_dir, args.lib,
            )

        pair_frame = common.JobSlice(pair_frame, args, logger)
        return pair_frame

    df = make_jobs(args)
    runner(df, args)
