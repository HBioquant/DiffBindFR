# Copyright (c) MDLDrugLib. All rights reserved.
from .misc import (check_prerequisites, concat_list, deprecated_api_warning,
                   import_modules_from_strings, is_list_of, to_5tuple, has_method,
                   is_method_overridden, is_seq_of, is_str, is_tuple_of,
                   iter_cast, list_cast, require_executable, require_package,
                   slice_lists, to_1tuple, to_2tuple, to_3tuple, to_4tuple,
                   to_ntuple, tuple_cast, color, NoneWithMessage)
from .version_utils import get_git_hash, digit_version
from .path import check_file_exist, mkdir_or_exists, is_filePath, symlink, fopen, find_vcs_root, scandir, search_dir_files
from .config import Config, ConfigDict, DictAction
from .config_utils import update_cfg_data_root, compat_cfg, replace_cfg_vals
from .file import get_line_count, extract, download, tmpdir_manager
from .timer import Timer, check_time, time_limit, timing, to_date
from .progressbar import track_progress, track_parallel_progress, track_iter_progress, ProgressBar
from .io import (input_choice, literal_eval, BaseStorageBackend, FileClient, list_from_file,
                dict_from_file, load, dump, register_handler)

try:
    import torch
except ImportError:
    __all__ = ['check_prerequisites', 'concat_list', 'deprecated_api_warning', 'get_git_hash', 'digit_version',
               'import_modules_from_strings', 'is_list_of', 'is_method_overridden',
               'is_seq_of', 'is_str', 'is_tuple_of', 'iter_cast', 'list_cast', 'has_method',
               'require_executable', 'require_package', 'slice_lists', 'to_1tuple', 'NoneWithMessage',
               'to_2tuple', 'to_3tuple', 'to_4tuple', 'to_5tuple', 'to_ntuple', 'tuple_cast',
               'get_line_count', 'extract', 'download', 'Config', 'ConfigDict', 'DictAction',
               'is_filePath', 'fopen', 'check_file_exist','mkdir_or_exists', 'symlink',
               'find_vcs_root', 'scandir', 'Timer', 'check_time', 'time_limit', 'color', 'input_choice',
               'literal_eval', 'ProgressBar', 'track_progress', 'track_parallel_progress',
               'track_iter_progress', 'BaseStorageBackend', 'FileClient', 'list_from_file',
               'dict_from_file', 'load', 'dump', 'register_handler', 'update_cfg_data_root',
               'compat_cfg', 'replace_cfg_vals', 'search_dir_files', 'timing', 'to_date', 'tmpdir_manager',

]
else:
    from .logger import get_logger, print_log, BaseRLLogger, RLLoggerPlaceholder, TensorboardRLLogger, WandbRLLogger
    from .registry import Registry, build_from_cfg
    from .parrots_wrapper import (
        TORCH_VERSION, BuildExtension, CppExtension, CUDAExtension, DataLoader,
        PoolDataLoader, SyncBatchNorm, _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd,
        _AvgPoolNd, _BatchNorm, _ConvNd, _ConvTransposeMixin, _InstanceNorm,
        _MaxPoolNd, get_build_config, is_rocm_pytorch, _get_cuda_home)
    from .trace import is_jit_tracing
    from .parrots_jit import jit, skip_no_elena
    from .hub import load_url
    __all__ = ['check_prerequisites', 'concat_list', 'deprecated_api_warning', 'get_git_hash', 'digit_version',
               'import_modules_from_strings', 'is_list_of', 'is_method_overridden',
               'is_seq_of', 'is_str', 'is_tuple_of', 'iter_cast', 'list_cast', 'has_method',
               'require_executable', 'require_package', 'slice_lists', 'to_1tuple', 'NoneWithMessage',
               'to_2tuple', 'to_3tuple', 'to_4tuple', 'to_5tuple', 'to_ntuple', 'tuple_cast',
               'get_logger', 'print_log', 'get_line_count', 'extract', 'download', 'skip_no_elena',
               'Config', 'ConfigDict', 'DictAction', 'is_filePath', 'fopen', 'check_file_exist',
               'mkdir_or_exists', 'symlink', 'find_vcs_root', 'scandir', 'Registry', 'build_from_cfg',
               'Timer', 'check_time', 'time_limit', 'color', 'input_choice', 'literal_eval', 'BaseRLLogger', 'jit',
               'RLLoggerPlaceholder', 'TensorboardRLLogger', 'WandbRLLogger', 'track_progress',
               'track_parallel_progress', 'track_iter_progress', 'ProgressBar','TORCH_VERSION', 'BuildExtension',
               'CppExtension', 'CUDAExtension', 'DataLoader', 'PoolDataLoader', 'SyncBatchNorm', '_AdaptiveAvgPoolNd',
               '_AdaptiveMaxPoolNd', '_AvgPoolNd', '_BatchNorm', '_ConvNd', '_ConvTransposeMixin', '_InstanceNorm',
               '_MaxPoolNd', 'get_build_config', 'is_rocm_pytorch', '_get_cuda_home',  'is_jit_tracing', 'load_url',
               'BaseStorageBackend', 'FileClient', 'list_from_file', 'dict_from_file', 'load', 'dump', 'register_handler',
               'update_cfg_data_root', 'compat_cfg', 'replace_cfg_vals', 'search_dir_files', 'timing', 'to_date', 'tmpdir_manager',
               ]