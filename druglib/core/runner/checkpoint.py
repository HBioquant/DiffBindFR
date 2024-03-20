# Copyright (c) MDLDrugLib. All rights reserved.
import io, os, re, time, warnings, pkgutil, logging, glob
import os.path as osp
from typing import Union, Optional, Any, Callable, List, Tuple
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
from torch.optim import Optimizer

import druglib
from druglib.utils import FileClient, load_url, mkdir_or_exists
from druglib.utils import print_log
from .dist_utils import get_dist_info
from .parallel import is_module_wrapper

ENV_MDLDRUGLIB_HOME = 'MDLDRUGLIB_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'

def _get_mdldruglib_home() -> str:
    mdldruglib_home = osp.expanduser(
        os.getenv(ENV_MDLDRUGLIB_HOME,
            osp.join(os.getenv(ENV_XDG_CACHE_HOME,
                DEFAULT_CACHE_DIR), 'mdldruglib')))

    mkdir_or_exists(mdldruglib_home)
    return mdldruglib_home

def load_state_dict(
        module:nn.Module,
        state_dict:OrderedDict,
        strict:bool = False,
        logger:Optional[logging.Logger] = None,
) -> None:
    """
    Load state_dict to a module.
    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    Args:
        module:nn.Module: Module that receives the state_dict.
        state_dict:OrderedDict: Weights.
        strict:bool: whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: `False`.
        logger:obj:`logging.Logger`, optional: Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)

def get_torchvision_models() -> dict:
    import torchvision
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


class CheckpointLoader(object):
    """
    A general checkpoint loader to manage all schemes.
    """
    _schemes = {}

    @classmethod
    def _register_scheme(
            cls,
            prefixes:Union[str, List[str], Tuple[str]],
            loader:Callable,
            force:bool = False
    ) -> None:
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f'{prefix} is already registered as a loader backend, '
                    'add "force=True" if you want to override it'
                )
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(sorted(cls._schemes.items(), key = lambda t:t[0], reverse = True))

    @classmethod
    def register_scheme(
            cls,
            prefixes:Union[str, List[str], Tuple[str]],
            loader:Optional[Callable] = None,
            force:bool = False,
    ) -> Optional[Callable]:
        """
        Register a loader to CheckpointLoader.
        This method can be used as a normal class method or a decorator.
        Args:
            prefixes:Union[str, List[str], Tuple[str]]: The prefix of the registered loader.
            loader:Optional[Callable]: The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force:bool: Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """
        if loader is not None:
            cls._register_scheme(prefixes, loader, force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(
            cls,
            path:str,
    ) -> Callable:
        """
        Finds a loader that supports the given path. Falls back to the local
            loader if no other loader is found.
        Args:
            path (str): Checkpoint path
        Returns:
            Callable: Checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix. For example, both 's3://path' and
            # 'open-mmlab:s3://path' should return `load_from_petrel`
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(
            cls,
            filename:str,
            map_location:Any = None,
            logger:Optional[logging.Logger] = None,
    ) -> Union[dict, OrderedDict]:
        """
        load checkpoint through URL scheme path.
       Args:
           filename:str: checkpoint file name with given prefix
           map_location:Any: Same as :func:`torch.load`.
               Defaults to None
           logger:Optional[logging.Logger]: The logger for message.
               Defaults to None
       Returns:
           dict or OrderedDict: The loaded checkpoint.
       """
        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        # This requires checkpoint_loader must be named "load_from_***"
        print_log(f'load checkpoint from {class_name[10:]} path: {filename}', logger)
        return checkpoint_loader(filename, map_location)


@CheckpointLoader.register_scheme(prefixes='')
def load_from_local(
        filename:str,
        map_location:Any,
) -> Union[dict, OrderedDict]:
    """
    Load checkpoint by local file path.
    Args:
        filename:str: Local checkpoint file path
        map_location:Any: Same as :func:`torch.load`.
    Returns:
        Union[dict, OrderedDict]: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes = ('http://', 'https://'))
def load_from_http(
        filename:str,
        map_location:Any = None,
        model_dir:Optional[str] = None
) -> Optional[Union[dict, OrderedDict]]:
    """
    Load checkpoint through HTTP or HTTPS scheme path. In distributed
        setting, this function only download checkpoint at local rank 0.
    Args:
        filename:str: Checkpoint file path with modelzoo or
            torchvision prefix
        map_location:Any: Same as :func:`torch.load`.
        model_dir:Optional[str]: Directory in which to save the object,
            Defaults to None

    Returns:
        Union[dict, OrderedDict] or None (if TORCH_VERSION < 1.7.0): The loaded checkpoint.
    """
    rank, world_size = get_dist_info()
    if rank == 0:
        checkpoint = load_url(filename, model_dir=model_dir, map_location=map_location)
    if world_size > 1:
        from torch import distributed
        distributed.barrier()
        if rank > 0:
            checkpoint = load_url(filename, model_dir=model_dir, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes='pavi://')
def load_from_pavi(
        filename:str,
        map_location:Any = None
) -> Union[dict, OrderedDict]:
    """
    Load checkpoint through the file path prefixed with pavi. In distributed
        setting, this function download ckpt at all ranks to different temporary
        directories.
    Args:
        filename:str: Checkpoint file path with pavi prefix.
        map_location:Any: Same as :func:`torch.load`.
          Defaults to None
    Returns:
        Union[dict, OrderedDict]: The loaded checkpoint.
    """
    assert filename.startswith('pavi://'), \
        f'Expected filename startswith `pavi://`, but get {filename}'
    model_path = filename[7:]

    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError(
            'Please install pavi to load checkpoint from modelcloud.')

    model = modelcloud.get(model_path)
    with TemporaryDirectory() as tmp_dir:
        downloaded_file = osp.join(tmp_dir, model.name)
        model.download(downloaded_file)
        checkpoint = torch.load(downloaded_file, map_location=map_location)
    return checkpoint

@CheckpointLoader.register_scheme(prefixes=r'(\S+\:)?s3://')
def load_from_petrel(
        filename:str,
        map_location:Any = None
) -> Union[dict, OrderedDict]:
    """
    Load checkpoint through the file path prefixed with s3.  In distributed setting,
        this function download ckpt at all ranks to different temporary directories.
    Note:
        The registered scheme prefixes have been enhanced to support bucket names
            in the path prefix, e.g. 's3://xx.xx/xx.path', 'bucket1:s3://xx.xx/xx.path'.
    Args:
        filename:str: Checkpoint file path with s3 prefix
        map_location:Any: Same as :func:`torch.load`.
    Returns:
        Union[dict, OrderedDict]: The loaded checkpoint.
    """
    file_client = FileClient(backend='petrel')
    with io.BytesIO(file_client.get(filename)) as buffer:
        checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=('modelzoo://', 'torchvision://'))
def load_from_torchvision(
        filename: str,
        map_location: Any = None
) -> Union[dict, OrderedDict]:
    """
    Load checkpoint through the file path prefixed with modelzoo or torchvision.
    Args:
        filename:str: Checkpoint file path with modelzoo or
            torchvision prefix
        map_location:Any: Same as :func:`torch.load`.
    Returns:
        Union[dict, OrderedDict]: The loaded checkpoint.
    """
    model_urls = get_torchvision_models()
    if filename.startswith('modelzoo://'):
        warnings.warn(
            'The URL scheme of "modelzoo://" is deprecated, please '
            'use "torchvision://" instead', DeprecationWarning)
        model_name = filename[11:]
    else:
        model_name = filename[14:]
    return load_from_http(model_urls[model_name], map_location=map_location)


def _load_checkpoint(
        filename:str,
        map_location:Any = None,
        logger:logging.Logger = None
) -> Union[dict, OrderedDict]:
    """
    Load checkpoint from somewhere (modelzoo, file, url).
    Args:
        filename:str: Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str, optional): Same as :func:`torch.load`.
           Default: None.
        logger (:mod:`logging.Logger`, optional): The logger for error message.
           Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint. It can be either an
           OrderedDict storing model weights or a dict containing other
           information, which depends on the checkpoint.
    """
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)


def _load_checkpoint_with_prefix(
        prefix:str,
        filename:str,
        map_location:Any = None
) -> Union[dict, OrderedDict]:
    """
    Load partial pretrained model with specific prefix.
    Args:
        prefix:str: The prefix of sub-module.
        filename:str: Accept local filepath, URL, `torchvision://xxx`, `open-mmlab://xxx`.
        map_location:Any: Same as :func:`torch.load`. Defaults to None.
    Returns:
        Union[dict, OrderedDict]: The loaded checkpoint.
    """

    checkpoint = _load_checkpoint(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict

def load_checkpoint(
        model:nn.Module,
        filename:str,
        map_location:Any = None,
        strict:bool = False,
        logger:Optional[logging.Logger] = None,
        revise_keys:List[Tuple[str]] = [(r'^module\.', ''),],
        drop_keys:Optional[List[str]] = None,
        use_ema: bool = False,
) -> Union[dict, OrderedDict]:
    """
    Load checkpoint from a file or URI.
    Args:
       model:nn.Module: Module to load checkpoint.
       filename:str: Accept local filepath, URL, `torchvision://xxx`, `open-mmlab://xxx`.
       map_location:str: Same as :func:`torch.load`.
       strict:bool: Whether to allow different params for the model and checkpoint.
       logger:Optional[logging.Logger]: The logger for error message.
       revise_keys:List[Tuple[str], ...]: A list of customized keywords to modify the
           state_dict in checkpoint. Each item is a (pattern, replacement)
           pair of the regular expression operations. Defaults to strip
           the prefix 'module.' by [(r'^module\\.', '')].
       drop_keys:List[str, ...]: A list of customized keywords pattern to be removed
           from state_dict in checkpoint.
       use_ema: bool. Whether use ema weight.
    Returns:
       Union[dict, OrderedDict]: The loaded checkpoint.
   """
    checkpoint = _load_checkpoint(filename, map_location, logger)

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')

    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if drop_keys is not None and len(drop_keys) > 0:
        state_dict = {k: v for k, v in state_dict.items() if not any(re.match(p, k) for p in drop_keys)}

    if use_ema:
        state_dict = {re.sub(r'^ema_', '', k): v for k, v in state_dict.items() if re.match(r'^ema_', k)}

    # strip prefix of state_dict
    metadata = getattr(checkpoint, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k):v for k, v in state_dict.items()}
        )
    # keep metadata of state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def weights_to_cpu(
        state_dict:Union[dict, OrderedDict],
) -> Union[dict, OrderedDict]:
    """
    Copy a model state dict to cpu.
    Args:
        state_dict:Union[dict, OrderedDict]: Model weights on GPU.
    Returns:
        Union[dict, OrderedDict]: Model weights on CPU.
    """
    state_dict_cpu = OrderedDict()
    for k, v in state_dict.items():
        state_dict_cpu[k] = v.cpu()
    state_dict_cpu._metadata = getattr(state_dict, '_metadata', OrderedDict())
    return state_dict_cpu

def _save_to_state_dict(
        module:nn.Module,
        destination:dict,
        prefix:str,
        keep_vars:bool,
) -> None:
    """
    Saves module state to `destination` dictionary.
    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module:nn.Module: The module to generate state_dict.
        destination:dict: A dict where state will be stored.
        prefix:str: The prefix for parameters and buffers used in this module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(
        module:nn.Module,
        destination:Optional[OrderedDict] = None,
        prefix:str = '',
        keep_vars:bool = False,
) -> Union[dict, OrderedDict]:
    """
    Returns a dictionary containing a whole state of the module.
    Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
    This method is modified from :meth:`torch.nn.Module.state_dict` to
        recursively check parallel module in case that the model has a complicated
        structure, e.g., nn.Module(nn.Module(DDP)).
    Args:
        module:nn.Module: The module to generate state_dict.
        destination:Optional[OrderedDict]: Returned dict for the state of the module.
        prefix:str: Prefix of the key.
        keep_vars:bool: Whether to keep the variable property of the
            parameters. Defaults to False.
    Returns:
        Union[dict, OrderedDict]: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_module_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(child, destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def save_checkpoint(
        model:nn.Module,
        filename:str,
        optimizer:Optional[Optimizer] = None,
        meta:Optional[dict] = None,
        file_client_args:Optional[dict] = None
) -> None:
    """
    Save checkpoint to file.
    The checkpoint will have 3 fields: `meta`, `state_dict` and
        `optimizer`. By default `meta` will contain version and time info.
    Args:
        model:nn.Module: Module whose params are to be saved.
        filename:str: Checkpoint filename.
        optimizer:Optional[Optimizer]: Optimizer to be saved.
        meta:Optional[dict]: Metadata to be saved in checkpoint.
        file_client_args:Optional[dict]: Arguments to instantiate a FileClient.
            Defaults to None.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(druglib_version = druglib.__version__, time = time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES = model.CLASSES)

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model))
    }
    # save optimizer state_dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    if filename.startswith('pavi://'):
        if file_client_args is not None:
            raise ValueError(
                'file_client_args should be "None" if filename starts with'
                f'"pavi://", but got {file_client_args}')
        try:
            from pavi import exception, modelcloud
        except ImportError:
            raise ImportError('Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except exception.NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        file_client = FileClient.infer_client(file_client_args, filename)
        with io.BytesIO() as f:
            torch.save(checkpoint, f)
            file_client.put(f.getvalue(), filename)


def find_latest_checkpoint(
        path: str,
        suffix: str = 'pth',
) -> Optional[str]:
    """
    Find the latest checkpoint from the working directory.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.
    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    ckpts = glob.glob(osp.join(path, f"*.{suffix}"))
    if len(ckpts) == 0:
        warnings.warn('The path of checkpoints does not exist.')
        return
    latest = -1
    checkpoint = None
    for ckpt in ckpts:
        order = int(osp.basename(ckpt).split('_')[-1].split('.')[0])
        if order > latest:
            latest = order
            checkpoint = ckpt
    return checkpoint