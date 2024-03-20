# Copyright (c) MDLDrugLib. All rights reserved.
# Reference by https://github.com/open-mmlab/mmcv/blob/master/mmcv/mmcv/utils/path.py
import os
import os.path as osp
from typing import Union, Sequence, Tuple, List
from pathlib import Path

def is_filePath(
        file:Union[str, Path]
) -> bool:
    return isinstance(file, str) or isinstance(file, Path)

def fopen(
        filepath:Union[str, Path],
        *args,
        **kwargs
):
  """
  A wrapped file open func.
  """
  if isinstance(filepath):
      return open(filepath, *args, **kwargs)
  elif isinstance(filepath, Path):
      return filepath.open(*args, **kwargs)
  raise ValueError('Args: `filepath` should be a string or Path ')

def check_file_exist(
        filename,
        msg_tmp = 'file "{}" does not exist.'
):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmp.format(filename))

def mkdir_or_exists(
        dirname,
        mode=0o777
):
    if dirname == '':
        return
    dirname = osp.expanduser(dirname)
    os.makedirs(dirname, mode=mode, exist_ok=True)


def symlink(
        src,
        dst,
        overwrite:bool = True,
        **kwargs
):
    if osp.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)

def find_vcs_root(
        path:str,
        markers:Sequence=('.git', )
):
    """
    Finds the root directory (including itself) of specified markers
    Args:
        path:str: Path of derectory or file.
        markers:Sequence: List or typle of file or directory name
    Returns:
        The directory contained one of the markers or None if not found.
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None

def scandir(
        dir:Union[str, Path],
        suffix:Union[str, Tuple[str], None] = None,
        recursive:bool=False
):
    """
    Scan a directory to find your interested file
    Args:
        dir:Union[str, Path]: Path of the directory.
        suffix:Union[str, Tuple[str], None]: File suffix that
            we are interested in. Defaults to None.
        recursive:bool: If set to be True, recursively scan the directory.
            Defaults to False.
    Returns:
        A generators for all the interested files with relative pathes.
    """
    if isinstance(dir, (str, Path)):
        dir = str(dir)
    else:
        raise TypeError('"dir" must be a string or Path')

    if suffix is not None and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of stings if you set a non-None.')

    root = dir

    def _scandir(
            dir,
            suffix,
            recursive
    ):
        for entry in os.scandir(dir):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None or rel_path.endswith(suffix):
                    yield rel_path
                elif recursive and osp.isdir(entry.path):
                    yield from _scandir(
                        entry.path, suffix, recursive
                    )
    return _scandir(dir, suffix, recursive)

def search_dir_files(
        dir: Union[str, Path],
        suffix: Union[str, Sequence[str], None] = None,
        exclude_dir: bool = False,
        exclude_file: bool = False,
) -> List[str]:
    """
    Search all files and dirs given root dir
    Args:
        dir: str or Path, root dir for searching;
        suffix: str, str of tuple, optional. File suffix to keep.
            If it is not specific (None), then all files will be included;
        exclude_dir: bool, whether exclude dirs in args `dir`;
            Default to Fasle, including all dirs.
        exclude_file: bool, whether exclude files in args `dir`;
            Default to False, including all files.
    E.g.
       /home/projects
            Model
                resnet.py
            Data
                PDBBind.pkl
            __init__.py
            test.py
        >>>  search_dir_files('/home/projects')
        ['Model', 'Model/resnet.py', 'Data', 'Data/PDBBind.pkl', '__init__.py', 'test.py']
        >>> search_dir_files('/home/projects', exclude_dir=True)
        ['Model/resnet.py', 'Data/PDBBind.pkl', '__init__.py', 'test.py']
        >>> search_dir_files('/home/projects', suffix='py', exclude_dir=True)
        ['Model/resnet.py', '__init__.py', 'test.py']
    So, user can easily concatenate the dir with the output relative file list to abspath.
    """
    files_list = []
    for root, dirs, files in os.walk(dir, topdown = True):
        relpath = osp.relpath(root, dir)
        if not exclude_dir:
            for name in dirs:
                files_list.append(osp.normpath(osp.join(relpath, name)))
        if not exclude_file:
            for name in files:
                if suffix is not None and not name.endswith(suffix):
                    continue
                files_list.append(osp.normpath(osp.join(relpath, name)))
    return files_list