# Copyright (c) MDLDrugLib. All rights reserved.
import os, warnings, re, copy, torch
from typing import Optional, Dict, Any
from collections.abc import MutableMapping
from argparse import Namespace
import numpy as np
from .logger import print_log
from .config import Config, ConfigDict


def update_cfg_data_root(
        cfg: Config,
        dst_root: Optional[str] = None,
):
    """Update data root according to env MMDET_DATASETS.

    If set env DRUGLIB_DATASETS, update cfg.data_root according to
    DRUGLIB_DATASETS. Otherwise, using cfg.data_root as default.

    Args:
        cfg (Config): The model config need to modify
    """
    assert isinstance(cfg, Config), f'cfg type error, expected Config, but got {type(cfg)}'
    if dst_root is None:
        if 'DRUGLIB_DATASETS' in os.environ:
            dst_root = os.environ['DRUGLIB_DATASETS']
            print_log(
                f'DRUGLIB_DATASETS has been set to be {dst_root}.'
                f'Using {dst_root} as data root.'
            )
        else:
            return
    else:
        assert isinstance(dst_root, str)

    # recursive update data root
    def update(cfg, src_root, dst_root):
        for k, v in cfg.items():
            if isinstance(v, Config):
                update(cfg[k], src_root, dst_root)
            if isinstance(v, str) and src_root in v:
                cfg[k] = v.replace(src_root, dst_root)

    update(cfg.data, cfg.data_root, dst_root)
    cfg.data_root = dst_root


def replace_cfg_vals(
        ori_cfg: Config
) -> Config:
    """
    Replace the string "${key}" with the corresponding value.

    Replace the "${key}" with the value of ori_cfg.key in the config. And
    support replacing the chained ${key}. Such as, replace "${key0.key1}"
    with the value of cfg.key0.key1. Code is modified from `vars.py
    < https://github.com/microsoft/SoftTeacher/blob/main/ssod/utils/vars.py>`_  # noqa: E501

    Args:
        ori_cfg (Config): The origin config with "${key}" generated from a file.

    Returns:
        updated_cfg [Config]: The config with "${key}" replaced by the corresponding value.
    """

    def get_value(cfg, key):
        for k in key.split('.'):
            cfg = cfg[k]
        return cfg

    def replace_value(cfg):
        if isinstance(cfg, dict):
            return {key: replace_value(value) for key, value in cfg.items()}
        elif isinstance(cfg, list):
            return [replace_value(item) for item in cfg]
        elif isinstance(cfg, tuple):
            return tuple([replace_value(item) for item in cfg])
        elif isinstance(cfg, str):
            # the format of string cfg may be:
            # 1) "${key}", which will be replaced with cfg.key directly
            # 2) "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx",
            # which will be replaced with the string of the cfg.key
            keys = pattern_key.findall(cfg)
            values = [get_value(ori_cfg, key[2:-1]) for key in keys]
            if len(keys) == 1 and keys[0] == cfg:
                # the format of string cfg is "${key}"
                cfg = values[0]
            else:
                for key, value in zip(keys, values):
                    # the format of string cfg is
                    # "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx"
                    assert not isinstance(value, (dict, list, tuple)), \
                        f'for the format of string cfg is ' \
                        f"'xxxxx${key}xxxxx' or 'xxx${key}xxx${key}xxx', " \
                        f"the type of the value of '${key}' " \
                        f'can not be dict, list, or tuple' \
                        f'but you input {type(value)} in {cfg}'
                    cfg = cfg.replace(key, str(value))
            return cfg
        else:
            return cfg

    # the pattern of string "${key}"
    pattern_key = re.compile(r'\$\{[a-zA-Z\d_.]*\}')
    # the type of ori_cfg._cfg_dict is mmcv.utils.config.ConfigDict
    updated_cfg = Config(
        replace_value(ori_cfg._cfg_dict), filename=ori_cfg.filename)
    # replace the model with model_wrapper
    if updated_cfg.get('model_wrapper', None) is not None:
        updated_cfg.model = updated_cfg.model_wrapper
        updated_cfg.pop('model_wrapper')
    return updated_cfg


def compat_cfg(cfg):
    """This function would modify some filed to keep the compatibility of
    config.

    For example, it will move some args which will be deprecated to the correct
    fields.
    """
    cfg = copy.deepcopy(cfg)
    cfg = compat_loader_args(cfg)
    cfg = compat_runner_args(cfg)
    return cfg


def compat_runner_args(cfg):
    if 'runner' not in cfg:
        cfg.runner = ConfigDict({
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        })
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    return cfg


def compat_loader_args(cfg):
    """Deprecated sample_per_gpu in cfg.data."""

    cfg = copy.deepcopy(cfg)
    if 'train_dataloader' not in cfg.data:
        cfg.data['train_dataloader'] = ConfigDict()
    if 'val_dataloader' not in cfg.data:
        cfg.data['val_dataloader'] = ConfigDict()
    if 'test_dataloader' not in cfg.data:
        cfg.data['test_dataloader'] = ConfigDict()

    # special process for train_dataloader
    if 'samples_per_gpu' in cfg.data:

        samples_per_gpu = cfg.data.pop('samples_per_gpu')
        assert 'samples_per_gpu' not in \
               cfg.data.train_dataloader, ('`samples_per_gpu` are set '
                                           'in `data` field and ` '
                                           'data.train_dataloader` '
                                           'at the same time. '
                                           'Please only set it in '
                                           '`data.train_dataloader`. ')
        cfg.data.train_dataloader['samples_per_gpu'] = samples_per_gpu

    if 'persistent_workers' in cfg.data:

        persistent_workers = cfg.data.pop('persistent_workers')
        assert 'persistent_workers' not in \
               cfg.data.train_dataloader, ('`persistent_workers` are set '
                                           'in `data` field and ` '
                                           'data.train_dataloader` '
                                           'at the same time. '
                                           'Please only set it in '
                                           '`data.train_dataloader`. ')
        cfg.data.train_dataloader['persistent_workers'] = persistent_workers

    if 'workers_per_gpu' in cfg.data:

        workers_per_gpu = cfg.data.pop('workers_per_gpu')
        cfg.data.train_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.val_dataloader['workers_per_gpu'] = workers_per_gpu
        cfg.data.test_dataloader['workers_per_gpu'] = workers_per_gpu

    # special process for val_dataloader
    if 'samples_per_gpu' in cfg.data.val:
        # keep default value of `sample_per_gpu` is 1
        assert 'samples_per_gpu' not in \
               cfg.data.val_dataloader, ('`samples_per_gpu` are set '
                                         'in `data.val` field and ` '
                                         'data.val_dataloader` at '
                                         'the same time. '
                                         'Please only set it in '
                                         '`data.val_dataloader`. ')
        cfg.data.val_dataloader['samples_per_gpu'] = \
            cfg.data.val.pop('samples_per_gpu')
    # special process for val_dataloader

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        if 'samples_per_gpu' in cfg.data.test:
            assert 'samples_per_gpu' not in \
                   cfg.data.test_dataloader, ('`samples_per_gpu` are set '
                                              'in `data.test` field and ` '
                                              'data.test_dataloader` '
                                              'at the same time. '
                                              'Please only set it in '
                                              '`data.test_dataloader`. ')

            cfg.data.test_dataloader['samples_per_gpu'] = \
                cfg.data.test.pop('samples_per_gpu')

    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            if 'samples_per_gpu' in ds_cfg:
                assert 'samples_per_gpu' not in \
                       cfg.data.test_dataloader, ('`samples_per_gpu` are set '
                                                  'in `data.test` field and ` '
                                                  'data.test_dataloader` at'
                                                  ' the same time. '
                                                  'Please only set it in '
                                                  '`data.test_dataloader`. ')
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        cfg.data.test_dataloader['samples_per_gpu'] = samples_per_gpu

    return cfg


def flatten_dict(
        d: Dict[Any, Any],
        delimiter: str = "@",
) -> Dict[str, Any]:
    """
    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        d: Dict[Any, Any], dictionary containing the hyperparameters
        delimiter: str, delimiter to express the hierarchy. Defaults to '@'.
    References:
        https://github.com/HannesStark/EquiBind/blob/main/commons/utils.py
    Returns:
        Flattened dict: Dict[str, Any]
    E.g.
        >>> flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """
    def _merge_keys(
            x: Any,
            prefixes: Optional[list] = None,
    ):
        prefixes = prefixes[:] if prefixes is not None else []
        if isinstance(x, MutableMapping):
            for k, v in x.items():
                k = str(k)
                if isinstance(v, (MutableMapping, Namespace)):
                    v = vars(v) if isinstance(v, Namespace) else v
                    for _d in _merge_keys(v, prefixes + [k]):
                        yield _d
                else:
                    yield prefixes + [k, v if v is not None else str(None)]

        else:
            yield prefixes + [x if x is None else str(x)]

    # merge keys-keys-...
    new_d = {delimiter.join(ks): v for *ks, v in _merge_keys(d)}

    # convert values
    for k, v in new_d.items():
        # convert relevant np scalars to python types first (instead of str)
        if isinstance(v, (np.bool_, np.integer, np.floating)):
            new_d[k] = v.item()
        elif not isinstance(v, (bool, int, float, str, torch.Tensor)):
            new_d[k] = str(v)

    return new_d