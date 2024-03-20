# Copyright (c) MDLDrugLib. All rights reserved.
import warnings, random, torch
from typing import Optional, Union
from functools import partial

import numpy as np
from torch.utils.data import DataLoader, Dataset
from druglib.data import collate

from druglib.utils import (TORCH_VERSION, Registry, build_from_cfg, digit_version,
                           Config)
from druglib.core import get_dist_info
from .samplers import (IterBatchSampler, IterGroupBatchSampler, DistributedSampler,
                       DistributedGroupSampler, GroupSampler)

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def default_worker_init_fn(
        work_id: int,
        num_workers: int,
        rank: int,
        seed: int,
):
    """worker seed initialization as the below formulation."""
    worker_seed = rank * num_workers + seed + work_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def random_worker_init_fn(
        work_id: int,
        num_workers: int,
        rank: int,
):
    """random worker seed initialization for random distribution sampling"""
    seed = torch.random.seed()
    worker_seed = rank * num_workers + seed + work_id
    worker_seed = worker_seed % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def build_dataset(
        cfg: Config,
        default_args: Optional[dict] = None,
):
    return build_from_cfg(cfg, DATASETS, default_args)


def build_dataloader(
        dataset: Union[Dataset],
        samples_per_gpu: int,
        workers_per_gpu: int,
        num_gpus: int = 1,
        dist: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
        runner_type: str = 'EpochBasedRunner',
        persistent_workers: bool = False,
        use_bgg: bool = False,
        **kwargs
) -> Union[DataLoader]:
    """
    Build PyTorch or PyTorch Geometric DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset: Union[Dataset, pyg_dataset], a PyTorch or PyTorch Geometric dataset.
        samples_per_gpu: int, number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu: int, how many subprocesses to use for data loading
            for each GPU.
        num_gpus: int, number of GPUs. Only used in non-distributed training. Default: 1.
        dist: bool, distributed training/test or not. Default: True.
        shuffle: bool, whether to shuffle the data at every epoch.
            Default: True.
        seed: int, Optional, seed to be used. Default: None.
        runner_type: str, type of runner. Default: `EpochBasedRunner`
        persistent_workers: bool, if True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers `Dataset` instances alive.
            This argument is only valid when PyTorch>=1.7.0. Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        dataloader: A PyTorch or PyTorch Geometric DataLoader.
    """
    follow_batch = kwargs.pop('follow_batch', None)
    exclude_keys = kwargs.pop('exclude_keys', None)

    rank, worldsize = get_dist_info()

    if dist:
        # When model is obj:`DistributedDataParallel`,
        # `batchsize` is obj:`dataloader` is the number
        # of training samples on each GPU.
        batchsize = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # When model is obj:`DataParallel`
        # `batchsize` is obj:`dataloader` is the all samples
        # on all GPUs
        batchsize = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    if runner_type == 'IterBasedRunner':
        # this is a batch sampler, which can yield
        # a mini-batch indices each time.
        # it can be used in both `DataParallel` and
        # `DistributedDataParallel`
        if shuffle:
            batchsampler = IterGroupBatchSampler(
                dataset,
                batchsize,
                rank,
                worldsize,
                seed,
                shuffle = True
            )
        else:
            batchsampler = IterBatchSampler(
                dataset,
                batchsize,
                rank,
                worldsize,
                seed,
                shuffle = False
            )
        batchsize = 1
        sampler = None
    else:
        if dist:
            # DistributedGroupSampler will definitely shuffle the data to
            # satisfy that images on each GPU are in the same group
            if shuffle:
                sampler = DistributedGroupSampler(
                    dataset,
                    samples_per_gpu,
                    rank,
                    worldsize,
                    seed
                )
            else:
                sampler = DistributedSampler(
                    dataset,
                    worldsize,
                    rank,
                    shuffle = shuffle,
                    seed = seed,
                )
        else:
            sampler = GroupSampler(
                dataset,
                samples_per_gpu
            ) if shuffle else None
        batchsampler = None

    init_fn = partial(
        default_worker_init_fn,
        num_workers = num_workers,
        rank = rank, seed = seed) if seed is not None \
        else partial(
        random_worker_init_fn,
        num_workers = num_workers, rank = rank,
    )

    if (TORCH_VERSION != 'parrots' and digit_version(TORCH_VERSION) >= digit_version('1.7.0')):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn('persistent_workers is invalid because your pytorch '
                      'version is lower than 1.7.0')

    if use_bgg:
        from prefetch_generator import BackgroundGenerator

        class _DataLoader(DataLoader):
            def __iter__(self):
                return BackgroundGenerator(super().__iter__())

        print('Use Background Generator supported dataloader.')
    else:
        _DataLoader = DataLoader

    dataloader = _DataLoader(
        dataset,
        batch_size = batchsize,
        sampler = sampler,
        batch_sampler = batchsampler,
        num_workers = num_workers,
        collate_fn = partial(
            collate,
            samples_per_gpu = samples_per_gpu,
            follow_batch = follow_batch,
            exclude_keys = exclude_keys,
        ),
        pin_memory = kwargs.pop("pin_memory", False),
        worker_init_fn = init_fn,
        **kwargs
    )

    return dataloader
