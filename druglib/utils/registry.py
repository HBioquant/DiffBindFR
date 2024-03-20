# Copyright (c) MDLDrugLib. All rights reserved.
import warnings
import inspect
from typing import Optional,Type

from .misc import is_seq_of

def build_from_cfg(
        cfg,
        registry,
        default_args:Optional[dict] = None
):
    """
    Build
    Args:
        cfg:dict: Config dict. It should at least contain the key "type".
        registry:`Registry`: The registry to search the type from.
    Return:
        object: The constructed objects

    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got type {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError('`cfg` or `default_args` must contain the key "type", '
                           f'but got cfg {cfg}\ndefault_args {default_args}.')
    if not isinstance(registry, Registry):
        raise TypeError('Registry must be an Registry object, but got {}'.format(type(registry)))
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError(
            'default_args must be either None or dict, but got defaults_args type {}'.format(type(default_args))
        )

    kargs = cfg.copy()

    if default_args is not None:
        for n, v in default_args.items():
            kargs.setdefault(n, v)

    obj_type = kargs.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'"{obj_type} is not in the "{registry.name} registry"'
            )
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'"type" in cfg must be a str or valid type, but got {type(obj_type)}.'
        )
    try:
        return obj_cls(**kargs)
    except Exception as e:
        raise type(e)(f'{obj_cls.__name__}: {e}')


class Registry:
    """
    A Registry to map strings to classes.

    Registered object could be built from registry.
    Reference from https://github.com/open-mmlab/mmcv/blob/master/mmcv/mmcv/utils/registry.py
    Args:
        name:str: Registry name
        build_func:func, optional: Build function to constuct instance from
            Registry, func:`build_from_cfg` is used if neither ``parent`` or
            ``build_func`` is specified. If ``parent`` is specified and
            ``build_func`` is not given, ``build_func`` will be inherited
            from ``parent``. Defualts to None.
        parent:Registry, optional: Parent registry. The class registry in children registry could be
            built from parent. Defaults to None.
        scope:str, optional: The scope of registry. It is the key to search for children registry. If
            not specified, scope will be the name if the package where class is defined.
            Defaults to None.
    E.g.:
        >>> MOLFINDER = Registry('molfinder')
        >>> @MOLFINDER.register_module()
        >>> class Transformer:
        >>>     pass
        >>> transfomer = MOLFINDER.build(dict(type='Transformer'))
    """
    def __init__(self,
                 name,
                 build_func=None,
                 parent = None,
                 scope = None
                 ):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = self.infer_scope() if scope is None else scope

        # self.build_func will be set with th following priorty:
        # 1. build_func
        # 2. parent.build_func
        # 3. build_from_cfg
        if build_func is None:
            if parent is not None:
                self.build_func = parent.build_func
            else:
                self.build_func = build_from_cfg
        else:
            self.build_func = build_func

        if parent is not None:
            assert isinstance(parent, Registry)
            parent._add_children(self)
            self.parent = parent
        else:
            self.parent = None

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(name = {self._name}, '\
            f'items = {self._module_dict})'
        return format_str

    @staticmethod
    def infer_scope():
        """
        Infer the scope of registry.
        The name of the package where registry is defined will be returned.
        E.g.:
            # in druglib/models/cnn/resnet.py
            >>> MODELS = Registry('models')
            >>> @MODELS.register_module()
            >>> class ResNet:
            >>>     pass
            The scope of ``ResNet`` will be ``druglib``.
            inspect.stack()[2][0] >> <frame at xxxxx, file '/data01/zhujintao/projects/druglib/models/cnn/resnet.py', line 10, code <module>>
            inspect.getmodule >> druglib.models.cnn.resnet
            split('.')[0] >> druglib
        Note: if you would like to move the registry file to more deeper or more shallow directory,
            there is no no any problem with the auto scope assign due to your fixed druglib module by __init__.py
        Returns:
              scope:str: The inferred scope name.
        """
        # inspect.stack() trace where this function is called, the index-2
        # indicates the frame where `infer_scope()` is called
        filename = inspect.getmodule(inspect.stack()[2][0]).__name__
        split_filename = filename.split('.')
        return split_filename[0]

    @staticmethod
    def split_scope_key(
            key:str
    ):
        """Split scope and key.
        The first scope will be split from key.

        E.g.:
            >>> Registry.split_scope_key('druglib.ResNet')
            'druglib', 'ResNet'
            >>> Registry.split_scope_key('HRNet')
            None, 'HRNet'
            >>> Registry.split_scope_key('druglib.utils.Registry')
            'druglib', 'utils.Registry'# Children will work
        Returns:
            scope:str,None: The first scope.
            key:str: The remaining key.
        """
        split_index = key.find('.')
        if split_index != -1:
            return key[:split_index + 1], key[split_index + 1:]
        return None, key

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def scope(self):
        return self._scope

    @property
    def children(self):
        return self._children

    def get(self, key):
        """
        Get a registered member although input would be [Scope].[real_key]
        key:str: The class name in the string format.
        Returns:
             class: The corresponding class.
        Hierarchy registry tree:
            You could also build modules from more than one OpenMMLab frameworks,
            e.g. you could use a rnn in druglib-1 for de novo, you may also combine
            an graph model in property prediction and transformer in ChemVision.
        All MODELS registries of the downstream codebases are children registry of druglib's MODELS.
        Basically, there are two ways to build a module from child or sibling registries.
        1. Build from children registries
        e.g.:
            In druglib-1, you define:
        #>>> from druglib.utils import Registry
        #>>> from druglib.utils.cnn import MODELS
        #>>> models = Registry('model', parent=MODELS)
        #>>> @models.registry_module()
        #>>> class NetA(nn.Module):
        #>>>     def forward(self,x):
        #>>>         return x
            In druglib-2, you define:# children registry
        #>>> from druglib.utils import Registry
        #>>> from druglib.cnn import MODELS
        #>>> models = Registry('model', parent=MODELS)
        #>>> @models.register_module()
        #>>> class NetB(nn.Module):
        #>>>     def forward(self, x):
        #>>>         return x + 1
            Then you can in druglib-1 codebase:#children registry
        #>>> from druglib-1.models import models
        #>>> net_a = models.build(cfg=dict(type='NetA'))
        #>>> net_b = models.build(cfg=dict(type='druglib-2.NetB'))
            Thus, you can in druglib-2 codebase:
        #>>> from druglib-2.models import models
        #>>> net_a = models.build(cfg=dict(type='druglib-1.NetA'))
        #>>> net_b = models.build(cfg=dict(type='druglib.NetB'))
            And, you even can in druglib codebase:# parent registry
        #>>> from druglib.cnn import MODELS
        #>>> net_a = MODELS.build(cfg=dict(type='druglib-1.NetA'))
        #>>> net_b = MODELS.build(cfg=dict(type='druglib-2.NetB'))
        """
        scope, real_key = self.split_scope_key(key)
        if scope is None or scope == self._scope:
            # get from self
            if real_key in self._module_dict:
                return self._module_dict[real_key]
        else:
            # get from self._children
            # multi-version {'druglib', 'druglib-1', 'druglib-2'}
            if scope in self._children:
                return self._children[scope].get(real_key)
            else:
                # get root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(key)

    def _add_children(self,
                      registry):
        """
        Add children for a registry.

        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.

        E.g.:
            >>> models = Registry('models')
            >>> druglib_models = Registry('models', parent=models)
            >>> @druglib_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(type = 'ResNet'))
        """
        assert isinstance(registry, Registry), '"regiostry" must be a Registry type.'
        assert registry.scope is not None
        assert registry.scope not in self.children, \
        f'scope {registry.scope} exists in {self.name} registry'
        self.children[registry.scope] = registry

    def build(self,
              *args,
              **kwargs):
        """
        Instantiate a someClass with cfg and default_args.
        Returns:
            a instantiated Class (__init__.py has already done by cfg and default_args)
        """
        return self.build_func(*args, **kwargs, registry=self)


    def register_module(self,
                        name:Optional[str]=None,
                        overwrite:bool=False,
                        module:Optional[Type]=None):
        """
        Registry a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as b=a decorator or a normal function.

        Args:
            name:Optional[str], option: The module name to be registry as a dict key.
                If not specified, the class name will be used.
            overwrite:bool, option: Whether to override an existing class with
                the same name. Defaults to False.
            module:Type: Module class to be registered.

        E.g.:
            >>> backbone = Registry('backbone')
            >>> @backbone.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbone = Registry('backbone')
            >>> @backbone.register_module(name='resnet')
            >>> class ResNet:
            >>>     pass
            >>> backbone = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbone.register_module(module=ResNet)
        """
        if not isinstance(overwrite, bool):
            raise KeyError(
                f'"overwrite" must be a boolean, but got a type "{type(overwrite)}"'
            )
        if not (isinstance(name, str) or name is None or is_seq_of(name, str)):
            raise KeyError(
                f'"name" must be either a string, None or a sequence if str, '
                f'but got a type "{type(name)}"'
            )

        # use it as a normal method: x.register_module(module = SomeClass)
        if module is not None:
            self._register_module(
                cls=module, name=name, overwrite=overwrite
            )
            return module

        def _register(cls):
            self._register_module(cls, name, overwrite)
            return cls

        return _register

    def _register_module(self,
                         cls,
                         name = None,
                         overwrite:bool = False):
        if not inspect.isclass(cls):
            raise TypeError('input cls must be a class, '
                            f'but got {type(cls)}')

        if name is None:
            name = cls.__name__
        if isinstance(name, str):
            name = [name]
        for n in name:
            if not overwrite and n in self._module_dict:
                raise KeyError(
                    f'{n} is already registered, in {self.name}'
                )
            self._module_dict[n] = cls

