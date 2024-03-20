# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, Union
import math
import itertools

import numpy as np
import torch
from torch.utils.data import Sampler

from druglib.core import get_dist_info, init_random_seed, get_device


class IterBatchSampler(Sampler):
    """
    Similar to `BatchSampler` warping a `DistributedSampler. It is designed
    iteration-based runners like `IterBasedRunner` and yields a mini-batch
    indices each time.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset: torch.utils.data.Dataset.
        samples_per_gpu: int, when model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU,
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        rank: int, optional, rank of current process. Default: None.
        num_replicas: int, optional, number of processes participating in
            distributed training. Default: None.
        seed: int, random seed. Default: 0.
        shuffle: bool, whether shuffle the dataset or not. Default: True.
    """
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            samples_per_gpu: int = 1,
            rank: Optional[int] = None,
            num_replicas: Optional[int] = None,
            seed: int = 0,
            shuffle: bool = True,
    ):
        _rank, _num_replicas = get_dist_info()
        if rank is None:
            rank = _rank
        if num_replicas is None:
            num_replicas = _num_replicas
        self.rank = rank
        self.num_replicas = num_replicas
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        device = get_device()
        self.seed = init_random_seed(seed, device, enable_sync = True)

        self.shuffle = shuffle
        self.datasize = len(self.dataset)
        self.indices = self._infinite_by_rank()

    def _infinite_indices(self):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.datasize, generator = g).tolist()
            else:
                yield from torch.arange(self.datasize).tolist()

    def _infinite_by_rank(self):
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(),
            self.rank,
            None,
            self.num_replicas,
        )

    def __iter__(self):
        """Once the data size is met, yield it"""
        buffer_indices = []
        for ind in self.indices:
            buffer_indices.append(ind)
            if len(buffer_indices) == self.samples_per_gpu:
                yield buffer_indices
                buffer_indices.clear()

    def __len__(self):
        """The length of sub-data size on per GPU"""
        return math.ceil(self.datasize / self.num_replicas)

    def set_epoch(self):
        """`set_epoch` not supported in `IterationBased` runner."""
        raise NotImplementedError

class IterGroupBatchSampler(IterBatchSampler):
    """
    The child of obj:`IterBatchSampler` for grouped batch sampling.
    Similar to `BatchSampler` warping a `GroupSampler. It is designed for
    iteration-based runners like `IterBasedRunner` and yields a mini-batch
    indices each time, all indices in a batch should be in the same group.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset: torch.utils.data.Dataset.
        samples_per_gpu: int, when model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU,
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        rank: int, optional, rank of current process. Default: None.
        num_replicas: int, optional, number of processes participating in
            distributed training. Default: None.
        seed: int, random seed. Default: 0.
        shuffle: bool, whether shuffle the dataset or not. Default: True.
    """
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            samples_per_gpu: int = 1,
            rank: Optional[int] = None,
            num_replicas: Optional[int] = None,
            seed: int = 0,
            shuffle: bool = True,
    ):
        super(IterGroupBatchSampler, self).__init__(
            dataset,
            samples_per_gpu,
            rank,
            num_replicas,
            seed,
            shuffle,
        )
        assert hasattr(self.dataset, "flag")
        self.flag = self.dataset.flag
        self.groupsize = np.bincount(self.flag)
        self.buffer_indices = {k:[] for k in range(len(np.bincount))}

    def __iter__(self):
        """Once the data size is met, yield it"""
        # TODO: if there exists rare group, it will be never yield
        for ind in self.indices:
            flag = self.flag[ind]
            buffer_list = self.buffer_indices[flag]
            buffer_list.append(ind)
            if len(buffer_list) == self.samples_per_gpu:
                yield buffer_list[:]
                del buffer_list[:]