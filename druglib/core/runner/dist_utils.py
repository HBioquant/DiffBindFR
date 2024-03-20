# Copyright (c) MDLDrugLib. All rights reserved.
import os, time, functools, subprocess, multiprocessing, warnings, socket, pickle
from collections import OrderedDict, defaultdict
from typing import Optional, Tuple, Union, Any, List, Callable, Iterable
import cv2
import numpy as np
import torch
from torch import Tensor
import torch.multiprocessing as tmp
from torch import distributed as dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                        _unflatten_dense_tensors)


def _find_free_port():
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def _is_free_port(port):
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)

def init_dist(
        launcher:str,
        backend:str = 'nccl',
        **kwargs,
) -> None:
    if tmp.get_start_method(allow_none=True) is None:
        tmp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(
        backend:str,
        **kwargs,
) -> None:
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    # TODO: use local_rank instead of rank % num_gpus
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend = backend,
        **kwargs,
    )


def _init_dist_mpi(
        backend:str,
        **kwargs,
) -> None:
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    dist.init_process_group(
        backend = backend,
        **kwargs,
    )


def _init_dist_slurm(
        backend: str,
        port: Optional[int] = None,
) -> None:
    """
    Initialize slurm distributed training environment.
    If argument `port` is not specified, then the master port will be system
        environment variable `MASTER_PORT`. If `MASTER_PORT` is not in system
        environment variable, then a default port `29500` will be used.
    Args:
        backend:str: Backend of torch.distributed.
        port:Optional[int]: Master port. Default to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f"scontrol show hostname {node_list} | head -n1"
    )
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass # use MASTER_PORT in the environment variable
    else:
        # if torch.distributed default port(29500) is available
        # then use it, else find a free port
        if _is_free_port(29500):
            os.environ['MASTER_PORT'] = '29500'
        else:
            os.environ['MASTER_PORT'] = str(_find_free_port())
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend = backend)


def get_rank() -> int:
    """
    Get the rank of this process in distributed processes.
    Returns:
        int: 0 or for single process case.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0

def get_world_size() -> int:
    """
    Get the total number of distributed processes.
    Returns:
        int: 1 for single process case.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1

def get_dist_info() -> Tuple[int, int]:
    return get_rank(), get_world_size()

def get_cpu_count() -> int:
    """
    Get the number of CPUs on this node.
    """
    return multiprocessing.cpu_count()

def synchronize() -> None:
    """
    Synchronize among all distributed processes.
    """
    if get_world_size() > 1:
        dist.barrier()

def set_cuda_visible_device(
        ngpus: int,
        total: int = 8,
):
    """Get unused GPU devices."""
    empty = []
    for i in range(total):
        command = 'nvidia-smi -i '+str(i)+' | grep "No running" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        if int(output) == 1:
            empty.append(i)

    if len(empty) < ngpus:
        warnings.warn('avaliable gpus are less than required')
        exit(-1)

    gpus = ','.join(map(str, empty))

    return gpus

def get_available_gpu(
        num_gpu: int = 1,
        min_memory: int = 1000,
        sample: int = 3,
        nitro_restriction: bool = True,
        verbose: bool = True,
):
    """
    Get available GPU for you, if you have 4 GPU, it will return the GPU with lowest memory usage.
    Args:
        num_gpu: number of GPU you want to use;
        min_memory: minimum memory;
        sample: number of sample;
        nitro_restriction: if True then will not distribute the last GPU for you;
        verbose: verbose mode.
    Return:
        str of best choices, e.g. '1, 2'.
    """
    sum = None
    for _ in range(sample):
        info = os.popen('nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv').read()
        info = np.array(
            [[id] + t.replace('%', '').replace('MiB','').split(',')
             for id, t in enumerate(info.split('\n')[1:-1])]).astype(np.int)
        sum = info + (sum if sum is not None else 0)
        time.sleep(0.2)

    avg = sum//sample
    if nitro_restriction:
        avg = avg[:-1]

    available = avg[np.where(avg[:,2] > min_memory)]
    if len(available) < num_gpu:
        warnings.warn('avaliable gpus are less than required')
        exit(-1)

    if available.shape[0] == 0:
        warnings.warn('No GPU available')
        return ''

    select = ', '.join(
        available[np.argsort(available[:, 1])[:num_gpu], 0].astype(np.str).tolist()
    )
    if verbose:
        print('Available GPU List')
        first_line = [['id', 'utilization.gpu(%)', 'memory.free(MiB)']]
        matrix = first_line + available.astype(np.int).tolist()
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        print('Select id #' + select + ' for you.')

    return select

def setup_multi_processes(
        cfg: dict,
):
    """
    Setup multi-processing environment variables
    """
    # set multi-process start method as `fork` to speed up the training
    mp_start_method = cfg.get('mp_start_method', 'spawn') # fork
    current_method = tmp.get_start_method(allow_none=True)
    if current_method is not None and current_method != mp_start_method:
        warnings.warn(
            f'Multi-processing start method `{mp_start_method}` is '
            f'different from the previous setting `{current_method}`.'
            f'It will be force set to `{mp_start_method}`. You can change '
            f'this behavior by changing `mp_start_method` in your config.')
        tmp.set_start_method(mp_start_method, force=True)
    # disable opencv multiprocessing to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    workers_per_gpu = cfg.data.get('workers_per_gpu', 1)
    if 'train_dataloader' in cfg.data:
        workers_per_gpu = max(cfg.data.train_dataloader.get('workers_per_gpu', 1), workers_per_gpu)

    if 'OMP_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

cpu_group = None
gpu_group = None

def get_group(
        device:torch.device
) -> dist.group:
    """
    Get the process group corresponding to the given device.
    Args:
        device:torch.device: query device.
    """
    group:Optional[dist.group] = cpu_group if device.type == 'cpu' else gpu_group
    if group is None:
        raise ValueError(
            f"{device.type.upper()} group is not initialized. "
            "Use init_process_group() to initialize it"
        )
    return group


def init_process_group(
        backend:str,
        init_method:Optional[str],
        **kwargs,
):
    """
    Initialize CPU and/or GPU process groups
    Args:
        backend:str: Communication backend. Use `nccl` for GPUs and `gloo` for CPUs.
        init_method:Optional[str]: URL specifying how to initialize the process group.
    """
    global gpu_group
    global cpu_group

    dist.init_process_group(backend, init_method = init_method, **kwargs)
    gpu_group = dist.group.WORLD
    if backend == "nccl":
        cpu_group = dist.new_group(backend = "gloo")
    else:
        cpu_group = gpu_group

def master_only(
        func:Callable
):
    """
    Only the main process run the input func
    E.g.
        @master_only()
        def yourcustomfunc()；
            ...
            return ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper

def _allreduce_coalesced(
        tensors:List[torch.Tensor],
        world_size:int,
        bucket_size_mb:int = -1,
) -> None:
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(
            tensors = tensors,
            size_limit = bucket_size_bytes,
        )
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
            bucket, _unflatten_dense_tensors(flat_tensors, bucket)
        ):
            tensor.copy_(synced)


def allreduce_params(
        params:list,
        coalesce:bool = True,
        bucket_size_mb:int = -1,
) -> None:
    """
    Allreduce parameters.
    Args:
        params:list: List of parameters or buffers of a model.
        coalesce:bool: Whether allreduce parameters as a whole.
            Default to True.
        bucket_size_mb:int: Size of bucket, the unit is MB.
            Default to -1.
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(
            params,
            world_size,
            bucket_size_mb
        )
    else:
        for tensor in params:
            dist.all_reduce(tensor.div_(world_size))# this means `data /= world_size`

def allreduce_grads(
        params:list,
        coalesce:bool = True,
        bucket_size_mb:int = -1,
) -> None:
    """
    Allreduce gradients.
    Args:
        params:list: List of parameters or buffers of a model.
        coalesce:bool: Whether allreduce parameters as a whole.
            Default to True.
        bucket_size_mb:int: Size of bucket, the unit is MB.
            Default to -1.
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    grads = [
        param.grad.data for param in params
        if param.requires_grad and param.grad is not None
    ]
    if coalesce:
        _allreduce_coalesced(
            params,
            world_size,
            bucket_size_mb
        )
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))

def _recursive_read(
        obj:Union[Tensor, dict, Iterable],
) -> Tuple[dict, dict]:
    values = defaultdict(list)
    sizes = defaultdict(list)
    if isinstance(obj, Tensor):
        values[obj.dtype] += [obj.flatten()]
        sizes[obj.dtype] += [torch.tensor([obj.numel()], device = obj.device)]
    elif isinstance(obj, dict):
        for v in obj.values():
            child_values, child_sizes = _recursive_read(v)
            for k, v in child_values.items():
                values[k] += v
            for k, v in child_sizes.items():
                sizes[k] += v
    elif isinstance(obj, Iterable):
        for v in obj:
            child_values, child_sizes = _recursive_read(v)
            for k, v in child_values.items():
                values[k] += v
            for k, v in child_sizes.items():
                sizes[k] += v
    else:
        raise TypeError(f"`torch.Tensor`, `dict`, `Iterable` "
                        f"type are expected, but got {type(obj)}.")

    return values, sizes

def _recursive_write(
        obj:Union[Tensor, dict, Iterable],
        values:dict,
        sizes:Optional[dict] = None
) -> Union[Tensor, dict, Iterable]:
    if isinstance(obj, Tensor):
        if sizes is None:
            size = torch.tensor([obj.numel()], device = obj.device())
        else:
            s = sizes[obj.dtype]
            size, s = s.split([1, len(s) - 1])
            sizes[obj.dtype] = s
        v = values[obj.dtype]
        new_obj, v = v.split([size, v.shape[-1] - size], dim = -1)
        # compatible with reduce / stack / cat
        new_obj = new_obj.view(new_obj.shape[:-1] + (-1, ) + obj.shape[1:])
        values[obj.dtype] = v
        return new_obj, values
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            new_obj[k], values = _recursive_write(v, values, sizes)
    elif isinstance(obj, Iterable):
        new_obj = []
        for v in obj:
            sub_new_obj, values = _recursive_write(v, values, sizes)
            new_obj.append(sub_new_obj)
    else:
        raise TypeError(f"`torch.Tensor`, `dict`, `Iterable` "
                        f"type are expected, but got {type(obj)}.")

    return new_obj, values

def reduce_mean(
        tensor: torch.Tensor,
):
    """Get the mean of tensor on different GPUs."""
    if not (dist.is_initialized() and dist.is_available()):
        return tensor

    tensor = tensor.clone()
    worldsize = get_world_size()
    dist.all_reduce(tensor.div_(worldsize), op = dist.ReduceOp.SUM)
    return tensor

def obj2tensor(
        pyobj,
        device = 'cuda',
):
    """Serialize picklable python object to tensor."""
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)

def tensor2obj(
        tensor,
) -> Any:
    """Deserialize tensor to picklable python object."""
    return pickle.loads(tensor.cpu().numpy().tobytes())

def all_reduce_dict(
        py_dict: dict,
        op: str = 'sum',
        to_float: bool = True,
) -> dict:
    """
    Apply all reduce function for python dict object.

    NOTE: make sure that py_dict in different ranks has the same keys and
        the values should be in the same shape.

    Args:
        py_dict: dict, Dict to be applied all reduce op.
        op: str,oOperator, could be 'sum' or 'mean'. Default: 'sum'
        to_float: bool, whether to convert all values of dict to float.
            Default: True.

    Returns:
        OrderedDict: reduced python dict object.
    """
    rank, worldsize = get_dist_info()
    if worldsize == 1:
        return py_dict
    assert isinstance(py_dict, dict)
    assert op in ['sum', 'mean'], f"args `op` must be one of 'sum' and 'mean', but got {op}."

    py_keys = list(py_dict.keys())
    if not isinstance(py_dict, OrderedDict):
        py_keys_tensor = obj2tensor(py_keys)
        dist.broadcast(py_keys_tensor, src = 0)
        py_keys = tensor2obj(py_keys_tensor)

    shape_list = [py_dict[k].shape for k in py_keys]
    numel_list = [py_dict[k].numel() for k in py_keys]

    if to_float:
        if rank == 0:
            warnings.warn('Note: the "to_float" is True, you need to '
                          'ensure that the behavior is reasonable.')
        flatten_tensor = torch.cat(
            [py_dict[k].flatten().float() for k in py_keys])
    else:
        flatten_tensor = torch.cat([py_dict[k].flatten() for k in py_keys])

    dist.all_reduce(flatten_tensor, op = dist.ReduceOp.SUM)
    if op == 'mean':
        flatten_tensor /= worldsize

    out_dict = {k: t.reshape(s)  for k, t, s in zip(py_keys, torch.split(flatten_tensor, numel_list), shape_list)}
    if isinstance(py_dict, OrderedDict):
        out_dict = OrderedDict(out_dict)

    return out_dict