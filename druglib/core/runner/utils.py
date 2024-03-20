# Copyright (c) MDLDrugLib. All rights reserved.
import os, random, sys, time, warnings, logging
from getpass import getuser
from socket import gethostname
from typing import Optional, Any
from argparse import Namespace
from functools import partial
from six.moves import map, zip
import numpy as np
import torch
import torch.distributed as dist
from .dist_utils import get_dist_info


def get_host_info() -> str:
    """
    Get hostname and username.
    Return empty string if exception raised, e.g. `getpass.getuser()` will
        lead to error in docker container.
    """
    host = ''
    try:
        host = f'{getuser()}@{gethostname()}'
    except Exception as e:
        warnings.warn(f'Host or user not found: {str(e)}')
    finally:
        return host

def get_time_str() -> str:
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def obj_from_dict(
        info:dict,
        parent:Optional[Namespace] = None,
        default_args:Optional[dict] = None,
) -> Any:
    """
    **Like utils.registry function build_from_cfg()**
    Initialize an object from dict.
    The dict must contain the key "type", which indicates the object type,
        it can be either a string or type, such as "list" or `list`. Remaining
        fields are treated as the arguments for constructing the object.
    Args:
        info:dict: Object types and arguments.
        parent:Optional[Namespace]: Module which may contain expected object classes.
        default_args:Optional[dict]: Default arguments for initializing the object.
    Returns:
        Any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'type' in info
    assert isinstance(default_args, dict) or default_args is None
    kargs = info.copy()
    obj_type = kargs.pop("type")
    if isinstance(obj_type, str):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError(
            f"type must be a str or valid type, but got {type(obj_type)}"
        )
    if default_args is not None:
        for k, v in default_args.items():
            kargs.setdefault(k, v)
    return obj_type(**kargs)

def init_random_seed(
        seed: Optional[int] = None,
        device: str = 'cuda',
        enable_sync: bool = False,
):
    """
    Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
        and then broadcast to all processes to prevent some potential bugs.
    This function considers two app cases:
    1. runner initialization: for all libraries random seed initialization.
        Specially, when trainer's args.diff_seed is set to True, then every rank
        have different seed, that is `enable_sync = False`;
    2. dataloader.sampler initialization: in distributed sampling, different ranks
        should sample non-overlapped data in the dataset. Therefore, this function
        is used to make sure that each rank shuffles the data indices in the same
        order based on the same seed. Then different ranks could use different indices
        to select non-overlapped data from the same data list. That is `enbale_sync = True`.

    Args:
        seed: int, Optional, the seed. Default to None.
        device: str, the device where the seed will be put on.
            Default to 'cuda'.
        enable_sync: bool, enable to make sure different ranks share the same seed.
            If True, this method is generally used in `DistributedSampler`,
            because the seed should be identical across all processes
            in the distributed group. All workers must call this function,
            otherwise it will deadlock.

    Returns:
        int: Seed to be used.
    """
    if seed is not None and not enable_sync:
        return seed
    if seed is None:
        # Make sure all ranks share the same random seed to prevent
        # some potential bugs.
        seed = np.random.randint(2 ** 31)

    rank, world_size = get_dist_info()

    assert isinstance(seed, int)

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)

    return random_num.item()

def set_random_seed(
        seed:Optional[int] = None,
        device: str = 'cuda',
        deterministic:bool = False,
        use_rank_shift:bool = False,
        logger: logging.Logger = None,
) -> int:
    """
    Set random seed.
    Args:
        seed:int: Seed to be used.
        device:str: 'cuda' or 'cpu'
        deterministic:bool: Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        use_rank_shift:bool: Whether to add rank number to the random seed to have different
            random seed in different threads. Default to False.
    """
    assert device in ['cuda', 'cpu']
    seed = init_random_seed(seed, device)
    if use_rank_shift:
        rank, _ = get_dist_info()
        seed += rank
    if logger is not None and isinstance(logger, logging.Logger):
        logger.info(
            f'`use_rank_shift` = {use_rank_shift},'
            f'Set random seed to {seed}, '
            f'deterministic: {deterministic}'
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

def multi_apply(
        func,
        *args,
        **kwargs,
):
    """
    Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments， Return must be tuple, list and other iterable type.
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def unmap(
        data,
        count,
        inds,
        fill = 0,
):
    """
    Unmap a subset of item (data) back to the original set of items (of size count)
    """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret

