# Copyright (c) MDLDrugLib. All rights reserved.
import os, logging, argparse
from typing import Optional, Union, List, Tuple, Any
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

import druglib
from druglib.utils import (
    Config, ProgressBar, get_logger,
    compat_cfg, replace_cfg_vals, update_cfg_data_root,
)
from druglib.utils.torch_utils import tensor_tree_map
from druglib.core.runner import (
    setup_multi_processes, get_device, set_random_seed,
    load_checkpoint, wrap_fp16_model, build_dp,
)
from druglib.models import build_task_model
from druglib.datasets import build_dataloader
from druglib.ops import smina_min_inplace, get_smina_score

import DiffBindFR
from .inference_dataset import InferenceDataset
from DiffBindFR.scoring import (
    Early_stopper, PassNoneDataLoader, KarmaDock
)

# error corrected tag
ec_tag = '_ec'

def move_to_cpu(data):
    fn = lambda x: x.detach().cpu()
    return tensor_tree_map(fn, data)


def load_cfg(
        args: argparse.Namespace,
        logger: Optional[logging.Logger],
) -> Config:
    info_fn = print if logger is None else logger.info

    cfg = Config.fromfile(str(args.config))

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to DRUGLIB_DATASETS
    update_cfg_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model:
        cfg.model.init_cfg = None

    cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()

    if vars(args).get('seed', None) is not None:
        set_random_seed(args.seed, logger = logger)

    if args.diffbindfr_pocket_radius is not None:
        old_pr = cfg.test_pre_transform_prot[1].cutoff
        cfg.test_pre_transform_prot[1].cutoff = args.diffbindfr_pocket_radius
        info_fn(f'Change default pocket radius {old_pr}A to {args.diffbindfr_pocket_radius}A.')

    return cfg

def load_dataloader(
        jobs_df: pd.DataFrame,
        args: argparse.Namespace,
        cfg: Config,
        logger: Optional[logging.Logger] = None,
        use_rdkit_conf: bool = False,
        batch_size: int = 16,
        batch_repeat: bool = False,
        use_bgg: bool = True,
        lmdb_mode: bool = True,
        distributed: bool = False,
        debug: bool = True,
) -> DataLoader:
    if logger is None:
        logger = get_logger(name = 'InferDataloader')

    root = args.export_dir / args.experiment_name
    root.mkdir(parents = True, exist_ok = True)

    logger.info(f'Start to prepare job (experiment name: '
                f'{args.experiment_name}).')
    inference_dataset = InferenceDataset(
        root = str(root),
        cfg = cfg.copy(),
        pair_frame = jobs_df,
        num_poses = args.num_poses,
        generate_multiple_conformer = use_rdkit_conf,
        default_processed = f'data', # save file to root/default_processed
        batch_repeat = batch_repeat, # duplicate index if batch size set to num_poses
        n_jobs = args.num_workers,
        verbose = args.verbose,
        chunk = 2000,
        remove_temp = True,
        lmdb_mode = lmdb_mode,
        debug = debug,
    )

    test_dataloader_default_args = dict(
        samples_per_gpu = batch_size,
        workers_per_gpu = args.num_workers,
        dist = distributed, shuffle = False,
        use_bgg = use_bgg,
    )

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(inference_dataset, **test_loader_cfg)

    return data_loader

def load_model(
        args: argparse.Namespace,
        cfg: Config,
        **kwargs
) -> torch.nn.Module:
    if args.checkpoint is None:
        args.checkpoint = Path(DiffBindFR.ROOT) / 'weights' / 'diffbindfr_paper.pth'
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    cfg.model.train_cfg = None
    model = build_task_model(cfg.model, test_cfg = cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    ld_ckpt = dict(
        map_location = kwargs.pop('map_location', 'cpu'),
        strict = kwargs.pop('strict', False),
        logger = kwargs.pop('logger', None),
        revise_keys = kwargs.pop('revise_keys', [(r'^module\.', ''), ]),
        drop_keys = kwargs.pop('drop_keys', None),
        use_ema = kwargs.pop('use_ema', False),
    )
    load_checkpoint(model, str(args.checkpoint), **ld_ckpt)
    model = build_dp(model, cfg.device, device_ids = cfg.gpu_ids, **kwargs)
    model.eval()

    return model

def model_run(
        dl: DataLoader,
        model: torch.nn.Module,
        show_traj: bool = False,
) -> List[Any]:
    results = []
    progbar = ProgressBar(task_num=len(dl.dataset))
    for idx, data in enumerate(dl):
        with torch.no_grad():
            batch_results = model(data, mode="test", visualize=show_traj)
            batch_results = move_to_cpu(batch_results)
        results.extend(batch_results)
        data_size = len(batch_results)
        for _ in range(data_size):
            progbar.update()

    return results

def inferencer(
        dl: DataLoader,
        model: torch.nn.Module,
        show_traj: bool = False,
        override: bool = False,
        out: Optional[Union[Path, str]] = None,
        logger: Optional[logging.Logger] = None,
) -> List[Tuple[Any, Any]]:
    if logger is None:
        logger = get_logger(name = 'Inferencer')

    if not override and (out is not None and Path(out).exists()):
        logger.info(f'Reload model inference output from {out}')
        pairs_results = druglib.load(str(out))
        return pairs_results

    dataset = dl.dataset
    timer = druglib.Timer()
    logger.info('Running model inference...')
    model_out = model_run(dl, model, show_traj)
    logger.info(f'Model inference is done which tasks {timer.since_start()}s')

    n_pairs = dataset.pair_frame.shape[0]
    num_poses = dataset.num_poses
    assert (n_pairs * num_poses) == len(model_out)
    pair_names = dataset.repeat_pair_names
    # slice the results to each pair
    if dataset.batch_repeat:
        results = [model_out[num_poses * i: num_poses * (i + 1)] for i in range(n_pairs)]
        pair_names = [pair_names[num_poses * i: num_poses * (i + 1)] for i in range(n_pairs)]
    else:
        results = [model_out[n_pairs * i: n_pairs * (i + 1)] for i in range(num_poses)]
        pair_names = [pair_names[n_pairs * i: n_pairs * (i + 1)] for i in range(num_poses)]
        results = list(zip(*results))
        pair_names = list(zip(*pair_names))
    pairs_results = list(zip(pair_names, results))

    if out is not None:
        logger.info(f'Export model results to path: {out}')
        druglib.dump(pairs_results, out)

    del model_out

    return pairs_results

def Scorer(
        test_dataset: torch.utils.data.Dataset,
        model_weight: Union[str, Path] = Path(DiffBindFR.ROOT) / 'weights' / 'mdn_paper.pt',
        output_path: Union[str, Path] = 'mdn_score.csv',
        batch_size: int = 16,
        device_id: Union[int, str] = 0,
        logger: Optional[logging.Logger] = None,
):
    if logger is None:
        logger = get_logger(name = 'Scorer')

    class DataLoaderX(PassNoneDataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())

    device = 'cpu'
    if torch.cuda.is_available():
        device = f'cuda:{device_id}'

    model = KarmaDock()
    model = nn.DataParallel(
        model,
        device_ids=[device_id],
        output_device=device_id,
    )
    model.to(device)

    stopper = Early_stopper(
        model_file=str(model_weight),
        mode='lower',
        patience=10,
    )

    logger.info('Load scoring model...')
    stopper.load_model(
        model_obj=model,
        my_device=device,
        strict=False,
        mine=True,
    )

    test_dataloader = DataLoaderX(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        follow_batch=[],
        pin_memory=True,
    )
    binding_scores = []
    with torch.no_grad():
        model.eval()
        for idx, data in enumerate(tqdm(test_dataloader)):
            data = data.to(device)
            batch_size = data['ligand'].batch[-1] + 1
            pro_node_s, lig_node_s = model.module.encoding(data)
            lig_pos = data['ligand'].xyz
            mdn_score_pred = model.module.scoring(
                lig_s=lig_node_s,
                lig_pos=lig_pos,
                pro_s=pro_node_s,
                data=data,
                dist_threhold=5.,
                batch_size=batch_size,
            )
            binding_scores.extend(mdn_score_pred.cpu().numpy().tolist())

        test_dataset.pair_frame['mdn_score'] = binding_scores
        test_dataset.pair_frame.to_csv(str(output_path), index=False)

    logger.info('Model Scoring is Done!')

    return

def error_corrector(
        docked_lig,
        score_only: bool = False,
        override: bool = False,
) -> float:
    work_dir = os.path.dirname(str(docked_lig))
    target_out = Path(work_dir) / f'lig_final{ec_tag}.sdf'
    if target_out.exists() and not override:
        return get_smina_score(str(target_out))

    score = smina_min_inplace(
        work_dir,
        rec_name = 'prot_final.pdb',
        lig_name = 'lig_final.sdf',
        out_name = f'lig_final{ec_tag}.sdf',
        score_only = score_only,
        exhaustiveness = None,
    )
    return score