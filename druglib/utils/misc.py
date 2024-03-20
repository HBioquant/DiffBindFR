# Copyright (c) MDLDrugLib. All rights reserved.
from typing import Optional, Union, Callable, List, Any
import collections.abc
import functools
import itertools
import subprocess
import warnings
from collections import abc
from importlib import import_module
from inspect import getfullargspec
from itertools import repeat
from dataclasses import dataclass


class color:
    """
    Colors for printing.
    """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# From PyTorch internals
def _ntuple(n:int):

    def parse(x) -> tuple:
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_5tuple = _ntuple(5)
to_ntuple = _ntuple

def is_str(x) -> bool:
    """
    Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)

def import_modules_from_strings(
        imports:Union[list, str, None],
        allow_failed_imports:bool = False
):
    """
    Import modules from the given list of strings.

    Args:
        import: list | str | None: The given module names to be imported.
        allow_failed_imports:bool: If True, the failed imports will return None.
            Otherwise, an ImportError is raise. Defaults to False.
    Returns:
        list[module] | module | None: The imported modules.
    e.g.:
        >>> osp, sys = import_modules_from_strings(
        ... ['os.path', 'sys']
        ... )
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_ | sys == == sys_
    """
    if not imports:
        return
    import_alone = False
    if isinstance(imports, str):
        import_alone = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'imports must be either list, str, None, but got type {type(imports)}'
        )
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannnot be imported. Str is needed.'
            )
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ingored.', UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if import_alone:
        imported = imported[0]

    return imported

def iter_cast(
        inputs:abc.Iterable,
        dst_type:type,
        return_type:Optional[type]=None
):
    """
    Cast elements of an iterable object into some type.
    Args:
        inputs:abc.Iterable: The input object.
        dst_type:type: Destination type.
        return_type:Optional[type]: if specified, the output object will be
            converted to this type, otherwise an iterator.
    Returns:
        iterator or specified return_type: The converted object
    """
    if not isinstance(inputs, abc.Iterable):
        raise TypeError(
            'Inputs must be an iterable object'
        )
    if not isinstance(dst_type, type):
        raise TypeError(
            '"dst_type" must be a valid type'
        )

    out_iterable = map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)

def list_cast(
        inputs: abc.Iterable,
        dst_type: type,
) -> list:
    """
    Cast elements of an iterable object into a list of some type.
    A partial method of :func: `iter_cast`
    Args:
        inputs:abc.Iterable: The input object.
        dst_type:type: Destination type.
    Returns:
        list: The converted object
    """
    return iter_cast(inputs, dst_type, return_type=list)

def tuple_cast(
        inputs: abc.Iterable,
        dst_type: type,
) -> tuple:
    """
    Cast elements of an iterable object into a tuple of some type.
    A partial method of :func: `iter_cast`
    Args:
        inputs:abc.Iterable: The input object.
        dst_type:type: Destination type.
    Returns:
        tuple: The converted object
    """
    return iter_cast(inputs, dst_type, return_type = tuple)

def is_seq_of(
        seq: abc.Sequence,
        expected_type: Union[type, tuple],
        seq_type: Optional[type] = None
) -> bool:
    """
    Check whether it is a sequence if some type.
    Args:
        seq:abc.Sequence: The sequence to be checked.
        expected_type:type: Expected type of sequence items.
        seq_type:Optional[tuple]: Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def is_list_of(
        seq: abc.Sequence,
        expected_type: type
) -> bool:
    """
    Check whether it is a list of some type.
    A partial method of :func:`is_seq_of`.
    Args:
        seq:abc.Sequence: The sequence to be checked.
        expected_type:type: Expected type of sequence items.
    Returns:
        bool: Whether the list is valid.
    """
    return is_seq_of(seq, expected_type, seq_type = list)

def is_tuple_of(
        seq: abc.Sequence,
        expected_type: type
) -> bool:
    """
    Check whether it is a tuple of some type.
    A partial method of :func:`is_seq_of`.
    Args:
        seq:abc.Sequence: The sequence to be checked.
        expected_type:type: Expected type of sequence items.
    Returns:
        bool: Whether the tuple is valid.
    """
    return is_seq_of(seq, expected_type, seq_type = tuple)

def slice_lists(
        in_lists: list,
        lens: Union[int, list]
) -> list:
    """
    Slice a list into several sub-lists by a list or int type of given length.

    Args:
        in_lists:list: The list to be sliced
        len:Union[int, list]: The expected length of each out list.
    Returns:
        list: A list of sliced list.
    """
    if isinstance(lens, int):
        assert len(in_lists) % lens == 0, f'Given lens ({lens} does not match list length ({in_lists})'
        lens = [lens] * (len(in_lists) // lens)
    if not isinstance(lens, list):
        raise TypeError(
            '"indices" of lens must be an matched integer of a list of list of integers.'
        )
    elif sum(lens) != len(in_lists):
        raise ValueError(
            'Sum of lens and length of list does not match: '
            f'{sum(lens)} != {len(in_lists)}'
        )
    out_lists = []
    idx = 0
    for i in range(len(lens)):
        out_lists.append(in_lists[idx:idx + lens[i]])
        idx += lens[i]
    return out_lists

def concat_list(
        in_list: list,
) -> list:
    """
    Concatenate a list of list into a single list.
    Args:
        in_list:list: The list of list to be merged.
    Returns:
        list: The concatenated flat list.
    """
    return list(itertools.chain(*in_list))

def _check_py_package(
        package: str,
) -> bool:
    try:
        import_module(package)
    except ImportError:
        return False
    else:
        return True

def _check_executable(
        cmd: str,
) -> bool:
    if subprocess.call(f'which {cmd}', shell=True) != 0:
        return False
    else:
        return True

def check_prerequisites(
        prerequisites: Union[List[str], str],
        checker: Callable,
        msg_tmpl: str='Prerequisites "{}" are required in method "{}" but not'
        'found, please install them first.'
):
    """
    A decorator factory to check if prerequisites are satisfied.

    prerequisites:Union[List[str], str]: Prerequisites to be checked.
    checker:Callable: The checker method that returns True if a prerequisite
        is meet, otherwise False.
    msg_tmpl:str: The message templates with two variable.

    Returns:
         decorator: A specified decorator.
    """

    def wrap(func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = [prerequisites] if isinstance(
                prerequisites, str
            ) else prerequisites
            missing = []
            for item in requirements:
                if not checker(item):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(', '.join(missing), func.__name__))
                raise RuntimeError('Prerequisites not meet.')
            else:
                return func(*args, **kwargs)

        return wrapped_func
    return wrap

def require_package(prerequisites):
    """
    A decorator to check if some python packages are installed.
    e.g.:
        >>> @ require_package('numpy')
        >>> def func(args1, args2):
        >>>     return numpy.zeros(1)
        array([0.])
        >>> @ require_package(['numpy', 'non-package'])
        >>> def func2(args1, args2):
        >>>     return numpy.ones(1)
        ImportError
    """
    return check_prerequisites(prerequisites, checker=_check_py_package)

def require_executable(prerequisites):
    """
    A decorator to check if some execuatble files are installed.
    e.g.:
        >>> @ require_executable('libsm')
        >>> def func(args1, args2):
        >>>     return numpy.zeros(1)
        1
    """
    return check_prerequisites(prerequisites, checker=_check_executable)

def deprecated_api_warning(
        name_dict:dict,
        cls_name:Optional[str] = None
):
    """
    A decorator to check if some arguments are deprecated and try to replace
    deprecated src_arg_name to dst_arg_name.

    Args:
        name_dict:dict:
            key:str: Deprecated argument names.
            value:str: Expected arguments names.
        cls_name:str: Consider class method function.
    Returns:
        func: New function.
    """
    def api_warning_wrapper(old_func):

        @ functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get name of old func
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f'{cls_name}.{func_name}'
            if args:
                arg_names = args_info.args[:len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead.'
                        )
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead.'
                        )
            # apply converted arguments to the decorated method
            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper

def is_method_overridden(
        method:str,
        base_class:type,
        derived_class:Union[Any, type],
) -> bool:
    """
    Check if a method of base class is override in derived class.

    Args:
        method:str: The method name to check.
        base_class:type: The class of the base class.
        derived_class:Union[type, Any]: The class of the derived class.
    """
    assert isinstance(base_class, type), \
    'base class does not accept instance, please pass class instead.'

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)

    return derived_method != base_method

def has_method(
        obj:object,
        method:str,
) -> bool:
    """
    Check whether the object has a method.
    Args:
        obj:object: The object to check.
        method:str: The method name to check.
    Returns:
        bool: True if the object has the methods else False
    """
    return hasattr(obj, method) and callable(getattr(obj, method))

@dataclass
class NoneWithMessage:
    message: str

    def __bool__(self):
        return False

    def __eq__(self, other):
        if other is None:
            return True
        return False

    def __repr__(self):
        return self.message