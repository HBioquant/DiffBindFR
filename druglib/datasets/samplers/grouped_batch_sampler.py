# Copyright (c) MDLDrugLib. All rights reserved.
import math, copy
from typing import Optional
import numpy as np
import torch
from druglib.core import get_dist_info
from torch.utils.data import Sampler, Dataset
from druglib.core import init_random_seed, get_device


class GroupSampler(Sampler):
    """
    Group mean data cluster type
    E.g. in the image task, you can set image aspect ratio
        as the cluster target, such as w / h > 1 as 0, and w / h
        < 1 as 1, then you can set dataset.flag as np.zero(len(dataset))
        and according  to the target, set the flag, so the sampler will
        shuffle according to the target.
    """
    def __init__(
            self,
            dataset: Dataset,
            samples_per_gpu: int = 1,
    ):
        assert hasattr(dataset, 'flag'), "Custom obj:`Dataset` needs `_set_group_flag when you call this sampler`"
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        flag = dataset.flag
        if isinstance(flag, torch.Tensor):
            flag = flag.cpu().numpy()
        self.flag = flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                np.random.shuffle(indice)
                num_extra = int(np.ceil(size / self.samples_per_gpu)
                                ) * self.samples_per_gpu - len(indice)
                indice = np.concatenate(
                    [indice, np.random.choice(indice, num_extra)])
                indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

class DistributedGroupSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`MDLDistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas: optional, number of processes participating in
            distributed training.
        rank: optional, rank of the current process within num_replicas.
        seed: int, optional, random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(
            self,
            dataset: Dataset,
            samples_per_gpu: int,
            rank: Optional[int] = None,
            num_replicas: Optional[int] = None,
            seed: Optional[int] = None,
    ):
        _rank, _worldsize = get_dist_info()
        if rank is None:
            rank = _rank
        if _worldsize is None:
            num_replicas = _worldsize

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.rank = rank
        self.num_replicas = num_replicas

        # enable seed sync.
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        device = get_device()
        self.seed = init_random_seed(seed, device, enable_sync = True) if seed is not None else 0

        # This indicates this sampler is epoch-based sampler.
        self.epoch = 0

        assert hasattr(dataset, 'flag')
        flag = dataset.flag
        if isinstance(flag, torch.Tensor):
            flag = flag.cpu().numpy()
        # predefined attributes
        self.flag = flag.astype(np.int64)
        self.group_size = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_size):
            # j / world_size = samples_per_gpu‘,
            # then samples_per_gpu' / samples_per_gpu -> ceil to padding.
            # allocate the sub-dataset in this rannk
            self.num_samples += int(math.ceil(j * 1.0 / self.num_replicas / self.samples_per_gpu )
                                     * self.samples_per_gpu)

        # all samples from different GPUs
        self.total_samples = self.num_samples * self.num_replicas

    def __iter__(self):
        # # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_size):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # shuffle the indices
                indice = indice[
                    list(torch.randperm(int(size), generator = g).numpy())
                ].tolist()
                num_extra = int(math.ceil( size * 1.0 / self.num_replicas / self.samples_per_gpu )
                                * self.num_replicas * self.samples_per_gpu) - size

                temp = copy.deepcopy(indice)
                for _ in range(num_extra // size):
                    indice.extend(temp)
                indice.extend(temp[:num_extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_samples


        # sub-dataset * world_size
        indices = [
            indices[j]
            for i in list(torch.randperm(len(indices) // self.samples_per_gpu, generator = g))
            for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)
        ]
        # a simpler method to flatten the shuffle list is applied by deepmind tree
        # import tree
        # indices = tree.flatten(indices)

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
