# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Mapping, Optional, Union, List
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

import torch_geometric as pyg
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData as PyGBaseData

from druglib.data import DataContainer


def collate(
        batch: Sequence,
        samples_per_gpu: int = 1,
        follow_batch: Optional[Union[List[str]]] = None,
        exclude_keys: Optional[Union[List[str]]] = None,
):
    """
    Puts each data field into a tensor/DataContainer with outer dimension
    batch size.
    in mmcv implement, ready for data formatting:
    1. formatting one after collect data by PIPELINES in training
        [{
            'img':DC(Tensor[C, H, W]),
            'gt_bbox':DC(Tensor[N,4]),
            'gt_label':DC(Tensor[N,]),
            'meta':DC({'flip':True, '..':bool}),
        }, ....] --> len(data) = Batch Size
    2. formatting two after MultiScaleFlipAug by PIPELINES in test
        [{
            'img':list[ num_augs * DC(Tensor[C, H, W]) ],
            'gt_bbox':list[ num_augs * DC(Tensor[N,4]) ],
            'gt_label':list[ num_augs * DC(Tensor[N,]) ],
            'meta':list[ num_augs * DC({'flip':True, '..':bool})],
        }, ....] --> len(data) = Batch Size
    in PyG implement, ready for data formatting:
        [{
            'ligand': DC(Graph),
            'protein': DC(Graph),
            'label': DC(Tensor(N,),
            'meta': DC({'rotate': True, '..': bool}),
        }
        ]
    Extend default_collate to add support for :type:`DataContainer`.
    There are 3 cases.
    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., data tensors, such as images tensors; PyG data, using batch
    3. cpu_only = False, stack = False, e.g., gt bboxes in cv area, constraints info in biology area
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    # data must be homogeneous
    batch_ele = batch[0]
    # if True, then batch \equiv [DC, DC, DC, DC, ...]
    if isinstance(batch_ele, DataContainer):
        stacked = []
        # TODO: this collate function is better suitable for cv task with image data
        #  and PyG Graph Learning tasl with image data,
        #  other tasks required new collate function.
        # cpu data, whether stacked or not, such as meta data
        if batch_ele.cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                # every element of stacked list has a list as [data, ...] --> len = samples_per_gpu
                # So len(stacked) = target_gpus
                stacked.append([bs.data for bs in batch[i:i + samples_per_gpu]])
            # Save `stacked` into DC
            return DataContainer(
                stacked, batch_ele.stack,
                batch_ele.padding_value,
                cpu_only = True,
                is_graph = False,
            )
        # return meta: DC(list[ num_gpus * list[samples_per_gpu * dict('flip','ori_shape'……)] ],stacked=False, cpu_only=True)
        # gpu data stacked, such as img data
        elif batch_ele.stack:
            if batch_ele.is_graph:
                # PyG Graph data-like implementation
                for i in range(0, len(batch), samples_per_gpu):
                    assert isinstance(batch_ele.data, PyGBaseData), "stacked data must be PyG Graph data"
                    stacked.append(
                        Batch.from_data_list(
                            data_list = [bs.data for bs in batch[i:i + samples_per_gpu]],
                            follow_batch = follow_batch,
                            exclude_keys = exclude_keys,
                        )
                    )
                return DataContainer(
                    stacked, batch_ele.stack,
                    cpu_only = False,
                    is_graph = True,
                )
            else:
                # cv image data-like implementation
                for i in range(0, len(batch), samples_per_gpu):
                    assert isinstance(batch_ele.data, torch.Tensor), "stacked data must be Tensor"
                    # pad_dims allows to be None or 1, 2, 3
                    if batch[i].pad_dims is not None:
                        ndim = batch[i].dim()
                        pad_dims = batch[i].pad_dims
                        assert ndim > pad_dims
                        # max shape temp
                        max_shape = [0 for _ in range(pad_dims)]
                        # initial max_shape
                        for dim in range(1, pad_dims + 1):
                            # if batch[i].data[C, H, W]
                            # if pad_dims = 1, max_shape == [W, ], len = 1
                            # if pad_dims = 2, max_shape == [W, H], len = 2
                            # if pad_dims = 3, max_shape == [W, H, C], len = 2
                            # reversely
                            max_shape[dim - 1] = batch[i].size(-dim)
                        # find max shape per axis
                        for sample in batch[i: i + samples_per_gpu]:
                            # forwardly
                            for dim in range(0, ndim - pad_dims):
                                assert sample.size(dim) == batch[i].size(0)
                            for dim in range(1, pad_dims + 1):
                                max_shape[dim - 1] = max(sample.size(-dim), max_shape[dim - 1])

                        padded_samples = []
                        for sample in batch[i: i + samples_per_gpu]:
                            pad = [0 for _ in range(pad_dims * 2)]
                            for dim in range(1, pad_dims +1):
                                pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                            padded_samples.append(
                                F.pad(
                                    sample.data, pad, value = sample.padding_value
                                )
                            )
                        stacked.append(
                            default_collate(padded_samples)
                        )

                    elif batch_ele.pad_dims is None:
                        # input to default_collate, return torch.stack results --> tensor(samples_per_gpu,C,H,W)
                        stacked.append(
                            default_collate([
                                sample.data for sample in batch[i:i + samples_per_gpu]
                            ])
                        )
                    # return with below DataContainer encapsulated:
                    # img: DC(list[ num_gpus * tensor(samples_per_gpu,C,H,W) ],stacked=True,cpu_only=False )

                    else:
                        raise ValueError('pad_dims should be either None or integers (1-3)')

        # gpu data unstacked, such as gt_bbox, gt_label
        else:
            for i in range(0, len(batch), samples_per_gpu):
                # every element of stacked list has a list as [data, ...] --> len = samples_per_gpu
                # So len(stacked) = target_gpus
                stacked.append([bs.data for bs in batch[i:i + samples_per_gpu]])
                # Save `stacked` into DC
                # return with below DataContainer encapsulated:
                # gt_bbox: DC(list[ num_gpus * list[samples_per_gpu * tensor] ],stacked=Fasle,cpu_only=False)
                # gt_label: DC(list[ num_gpus * list[samples_per_gpu * tensor] ], stacked=False,cpu_only=False)

        return DataContainer(
            stacked, batch_ele.stack,
            batch_ele.padding_value,
            cpu_only = False,
            is_graph = False,
        )

    elif isinstance(batch_ele, Sequence):
        # This is ready for secondly input formatting two,
        # formatting one will not pass this condition.
        # in this condition, input batch is [ list[ num_augs * DC(Tensor[C, H, W]) ], ... ] --> len = Batch Size
        transposed = zip(*batch)
        # after transposed, return [ list[ Batch Size * DC(Tensor[C, H, W]) ], ... ] --> len = num_augs
        # then every list[ Batch Size * DC(Tensor[C, H, W]) ] inputs,
        # just as formatting one does when input in the below Mapping condition
        # so return a dict == {key:list[ num_augs * DC[list[ num_gpus * list[ samples_per_gpu * tensor ]/Tensor[Batch, ...] ] ] ], ... }
        return [collate(samples, samples_per_gpu) for samples in transposed]

    elif isinstance(batch_ele, Mapping):
        # formatting one and two must be firstly input here
        # in mmcv, every key such as img will be put into a list, [DC(Tensor[C, H, W]), ....] --> len = Batch Size
        # in general, the list is [key_value1, key_value2, ...]  --> len = Batch Size
        # so return a dict == {key:DC[list[ num_gpus * list[ samples_per_gpu * tensor ]/Tensor[Batch, ...] ] ], ... }
        return {
            key: collate([b[key] for b in batch], samples_per_gpu) for key in batch_ele
        }

    else:
        return default_collate(batch)