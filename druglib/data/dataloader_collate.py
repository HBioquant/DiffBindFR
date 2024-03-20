# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Mapping, Optional, Union, List
from collections.abc import Sequence

from torch.utils.data.dataloader import default_collate
from .data_container import DataContainer
from .data import BaseData
from .batch import Batch


def collate(
        batch: Sequence,
        samples_per_gpu: int = 1,
        follow_batch: Optional[Union[List[str]]] = None,
        exclude_keys: Optional[Union[List[str]]] = None,
):
    """
    Puts each data field into a tensor/DataContainer with outer dimension
    batch size.
    in the implement, ready for data formatting:
    1. formatting one after collect data by PIPELINES in training
        [Data{
            'img':DC(Tensor[C, H, W]),
            'gt_bbox':DC(Tensor[N,4]),
            'gt_label':DC(Tensor[N,]),
            'metastore':DC({'flip':True, '..':bool}),
        }, ....] --> len(data) = Batch Size
    2. formatting two after MultiScaleFlipAug by PIPELINES in test
        [[
        num_augs * Data{"img": ...}
        ], ....] --> len(data) = Batch Size
    in PyG implement, ready for data formatting:
        [Data{
            'x': DC(Tensor [num_nodes, node_feature]),
            'edge_index': DC(Tensor [2, num_edges] or SparseTensor),
            'y': DC(Tensor [num_nodes,] or graph-level [1, ...]),
            'metastore': DC({'num_nodes': 10, '..': bool}),
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

    batch_ele = batch[0]
    if isinstance(batch_ele, BaseData):
        # TODO: this collate function is better suitable for cv task with image data
        #  and PyG Graph Learning tasl with image data,
        #  other tasks required new collate function.
        batch_list = []
        for i in range(0, len(batch), samples_per_gpu):
            batch_per_gpu = Batch.from_data_list(
                batch[i:i + samples_per_gpu],
                follow_batch = follow_batch,
                exclude_keys = exclude_keys
            )
            batch_list.append(batch_per_gpu)
        return DataContainer(
            data = batch_list,
        )

    elif isinstance(batch_ele, Sequence):
        transposed = zip(*batch)
        return [collate(
            samples,
            samples_per_gpu,
            follow_batch,
            exclude_keys
        ) for samples in transposed]

    elif isinstance(batch_ele, Mapping):
        return {
            key: collate(
                [b[key] for b in batch],
                samples_per_gpu,
                follow_batch,
                exclude_keys
            ) for key in batch_ele
        }

    else:
        return default_collate(batch)