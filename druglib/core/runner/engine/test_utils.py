# Copyright (c) MDLDrugLib. All rights reserved.
import os.path as osp
import time, pickle, shutil, tempfile
from typing import Optional
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

import druglib
from druglib import ProgressBar, mkdir_or_exists
from ..dist_utils import get_dist_info


def single_gpu_inference(
        model: nn.Module,
        data_loader: DataLoader,
) -> list:
    """
    Model inference with one single gpu.
    Args:
        model: nn.Module, model to be tested.
        data_loader: DataLoader.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    progbar = ProgressBar(task_num = len(dataset))
    # count = 0
    for data in data_loader:
        # count += 1
        # if count == 5:
        #     break
        with torch.no_grad():
            # Note that result can be any type
            result = model(data, mode = "test")
        results.extend(result)
        data_size = len(result)
        for _ in range(data_size):
            progbar.update()
    return results

def multi_gpu_inference(
        model: nn.Module,
        data_loader: DataLoader,
        collect_gpu: bool = False,
        tmpdir: Optional[str] = None,
) -> Optional[list]:
    """
    Model inference with multiple gpus.
    This function implements model inference with multiple gpus and hold two different
        results collections setting for user-defined purpose: gpu collections and cpu
        collections mode.
    In the gpu collections mode, results are encoded to tensor in gpus, and results collections
        are done by gpu communications;
    In the cpu collections mode, results are saved to the tempfile in every gpu, and root rank will
        execute the data collection task, though this is not elegant.
    Args:
        model: nn.Module, model to be tested.
        data_loader: DataLoader.
        collect_gpu: bool, results collection mode, either cpu or gpu mode. Default to False (Using cpu mode).
        tmpdir: Optional[str], cpu mode required temp directory.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, worldsize = get_dist_info()
    if rank == 0:
        progbar = ProgressBar(task_num = len(dataset))
    time.sleep(2)
    for i, data in enumerate(dataset):
        # every rank must iterat the same number of data.
        with torch.no_grad():
            result = model(data, mode = "test")
        results.extend(result)

        if rank == 0:
            data_size = len(result)
            overall = data_size * worldsize
            # if there exists batchsize inconsistency, it must be final iteration,
            # simply clip it
            if progbar.completed + overall > len(dataset):
                overall = len(dataset) - progbar.completed
            for _ in range(overall):
                progbar.update()# noqa

    if collect_gpu:
        return gpu_results_collections(results, len(dataset))
    else:
        return cpu_results_collections(results, len(dataset), tmpdir)

def gpu_results_collections(
        rankwise_results: list,
        data_size: int,
) -> Optional[list]:
    """
    Results collections with gpu mode
    Args:
        rankwise_results: list, containing result to be collected.
        data_size: int, the size of results, always equal to length of the results.
    Returns:
        Optional[list]: the collected results.
    """
    rank, worldsize = get_dist_info()
    # 1. dump rank-wise results to tensor with pickle for node communication
    rank_tensor = torch.tensor(
        bytearray(pickle.dumps(rankwise_results)), dtype = torch.uint8, device = "cuda",
    )
    # 2. gather all rank-wise results tensor shape (1D tensor)
    # 2.1 rank-wise 1D tensor length
    ts_shape = torch.tensor(rank_tensor.shape, device = "cuda")
    # 2.2 formatting tensor list for gather
    shape_list = [ts_shape.clone() for _ in range(worldsize)]
    # 2.3 now shape list includes gathered tensor shape different with th above equivalent shape_list
    dist.all_gather(shape_list, ts_shape)
    # 3. padding rank-wise tensor to max shape for gather results
    max_shape = torch.tensor(shape_list).max()
    send_tensor = torch.zeros(max_shape, dtype = torch.uint8, device = "cuda")
    send_tensor[:ts_shape[0]] = rank_tensor
    ts_list = [
        rank_tensor.new_zeros(max_shape) for _ in range(worldsize)
    ]
    dist.all_gather(ts_list, send_tensor)

    if rank == 0:
        # 4. ts_list [max_shape tensor * worldsize] -> [rank-wise shape tensor * worldsize]
        # pickle.loads -> merge_list [rank-wise results: [result-1, result-2, ...] * worldsize]
        merge_list = []
        for ts, shape in zip(ts_list, shape_list):
            rank_result = pickle.loads(ts[:shape[0]].cpu().numpy().tobytes())
            if rank_result:
                merge_list.append(rank_result)
        # sort the results
        ordered_results = []
        # TODO: risk of different batch number in different gpu?
        for res in zip(*merge_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        return ordered_results[:data_size]
    else:
        return None

def cpu_results_collections(
        rankwise_results: list,
        data_size: int,
        tmpdir: Optional[str] = None,
) -> Optional[list]:
    """
    Results collections with cpu mode
    Args:
        rankwise_results: list, containing result to be collected.
        data_size: int, the size of results, always equal to length of the results.
        tmpdir: Optional[str], temp directory for collected results to store. If set to None,
            it will create a random temporal directory for it.
    Returns:
        Optional[list]: the collected results.
    """
    rank, worldsize = get_dist_info()
    # create a tmp dir if not specified
    if tmpdir is None:
        MAX_LEN = 512
        default_tmpdir = ".dist_inference"
        # 32 is whitespace
        dir_ts = torch.full(
            (MAX_LEN, ),
            32,
            dtype = torch.uint8,
            device = "cuda",
        )
        if rank == 0:
            mkdir_or_exists(default_tmpdir)
            tmpdir = tempfile.mkdtemp(dir = default_tmpdir)
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype = torch.uint8, device = "cuda"
            )
            dir_ts[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_ts, 0)
        tmpdir = dir_ts.cpu().numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exists(tmpdir)
    # dump the part result to the dir
    rank_file = osp.join(tmpdir, f"rank_{rank}.pkl")
    druglib.dump(rankwise_results, rank_file)
    dist.barrier()

    if rank != 0:
        return None
    else:
        # load results of all ranks from tmpdir
        merge_list = []
        for i in range(worldsize):
            ts = druglib.load(osp.join(tmpdir, f"rank_{i}.pkl"))
            if ts:
                merge_list.append(ts)
        # sort the results
        ordered_results = []
        # TODO: risk of different batch number in different gpu?
        for res in zip(*merge_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:data_size]
        shutil.rmtree(tmpdir)
        return ordered_results