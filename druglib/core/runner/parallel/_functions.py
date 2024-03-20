# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Union, Optional, List, Tuple

import torch
from torch import Tensor
from torch.nn.parallel._functions import _get_stream
from druglib.data import BaseData
from druglib.utils import TORCH_VERSION, digit_version


def scatter(
        input: Union[List, Tensor],
        devices: List,
        streams: Optional[List] = None,
) -> Union[List, Tensor]:
    if streams is None:
        streams = [None] * len(devices)
    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        # scatter: distribute data to different gpus
        outputs = [scatter(
            input[i], [devices[i // chunk_size]], [streams[i // chunk_size]]
        ) for i in range(len(input))]
        return outputs
    elif isinstance(input, (Tensor, BaseData)):
        output = input.contiguous()
        stream = streams[0] if output.numel() > 0 else None

        if devices != [-1]:
            # when stream == None, no cpu to gpu works
            with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
                output = output.cuda(devices[0], non_blocking=False)
        return output
    else:
        raise Exception(f"Unknown type {type(input)}")

def get_input_device(
        input: Union[List, Tensor, BaseData],
) -> int:
    """-1 represents CPU; input device is either cpu or the same cuda id"""
    if isinstance(input, List):
        for t in input:
            device = get_input_device(t)
            if device != -1:
                return device
        return -1
    elif isinstance(input, (Tensor, BaseData)):
        return input.get_device() if input.is_cuda() else -1
    else:
        raise Exception(f"Unknown type {type(input)}")

def synchronize_stream(
        output: Union[List, Tensor, BaseData],
        devices: List,
        streams: List,
) -> None:
    if isinstance(output, List):
        chunk_size = len(output) // len(devices)
        for d in range(len(devices)):
            for c in range(chunk_size):
                synchronize_stream(
                    output[ d * chunk_size + c ],
                    [devices[d]],
                    [streams[d]]
                )
    elif isinstance(output, (Tensor, BaseData)):
        if output.numel() > 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception(f"Unknown type {type(output)}")

class Scatter:

    @staticmethod
    def forward(
            target_gpus: List[int],
            input: Union[Tensor, List, BaseData],
    ) -> Tuple:
        device = get_input_device(input)
        streams = None
        if device == -1 and target_gpus != [-1]:
            # Perform CPU to GPU copies in a background stream
            if digit_version(TORCH_VERSION) < digit_version('2.0.0'):
                streams = [_get_stream(gpu) for gpu in target_gpus]
            else:
                streams = [_get_stream(torch.device(f'cuda:{gpu}')) for gpu in target_gpus]
        outputs = scatter(input, target_gpus, streams)
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs) if isinstance(outputs, list) else (outputs, )
