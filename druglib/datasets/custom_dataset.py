# Copyright (c) MDLDrugLib. All rights reserved.
import logging, copy
import os.path as osp
from typing import (
    Optional, Union, List, Sequence,
    Tuple, Mapping, Callable, Any,
)
from abc import abstractmethod
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import druglib
from druglib.utils import color
from druglib.utils import (
    mkdir_or_exists,
    list_from_file,
    search_dir_files,
)
from .base_pipelines.compose import Compose

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class CustomDataset(Dataset):
    """
    Abstract custom dataset class for integrated task, such as cv and graph task.
    Args:
        root: str, optional. The root data directory, saving raw data and processing data
            at least two folders, in default, called :folder:'raw' and :folder:'processed'.
        pre_transform: Sequence[Mapping | Callable], optional. Used in :method:`processed`,
            act as preprocess data pipeline. Once the processed data has been generated,
            it will be useless.
        transform: Sequence[Mapping | Callable], optional. Used in :method:`get` to online
            process data.
        test_mode: bool. If True, it must be in the training or val mode; otherwise, it must be in
            the testing mode without annotations.
        classes: str, Sequence[str], optional. Classification task classes.
        raw_data_suffix: str, Sequence[str], optional. Work if args `raw_data` is None, and will
            search the raw dir for the all specified suffix files.
            E.g. '.pdb', then will search the raw dir for .pdb file in any folders with topdown-manner.
        raw_data: str, Sequence[str], optional. If specified, the raw_data_suffix will never work and
            directly return the args `raw_data` as the real raw data.
        If specified file is not found, it turns to download the raw data.
        processed_file_suffix: str, Sequence[str], optional. It has the same effect as args `raw_data_suffix`,
            but it's searching for processed data.
            E.g. '.pt', then will search the processed dir for .pt file in any folders with topdown-manner.
        processed_file: str, Sequence[str], optional. It has the same effect as args `raw_data`,
            but it returns processed data.
        If specified file is not found, it turns to process the raw data.
        **kwargs: Any other args for :method:`process`.
    """
    CLASSES: Union[str, List[str]] = None # Classification task needed

    defualt_raw: str = 'raw'
    url: Optional[str] = None

    default_processed: str = 'processed'

    def __init__(
            self,
            root: Optional[str] = None,
            pre_transform: Union[Sequence[Mapping], Sequence[Callable], None] = None,
            transform: Union[Sequence[Mapping], Sequence[Callable], None] = None,
            test_mode: bool = False,
            classes: Union[str, Sequence[str], None] = None,
            raw_data_suffix: Union[str, Sequence[str], None] = None,
            raw_data: Union[str, Sequence[str], None] = None,
            processed_file_suffix: Union[str, Sequence[str], None] = None,
            processed_file: Union[str, Sequence[str], None] = None,
            defualt_raw: Optional[str] = None,
            default_processed: Optional[str] = None,
            **kwargs
    ):
        super(CustomDataset, self).__init__()
        if druglib.is_str(root):
            root = osp.expanduser(osp.normpath(root))

        logger = kwargs.pop('log', None)
        if logger is None:
            logger = logging.getLogger(name=self.__class__.__name__)
        if not isinstance(logger, logging.Logger):
            raise TypeError(f'Dataset log must be None or logging.Logger, but got {type(logger)}')
        self.logger = logger
        self.logger.setLevel(level=logging.DEBUG if kwargs.get('debug', False) else logging.INFO)

        self.root = root
        self.pre_transform = Compose(pre_transform) if pre_transform is not None else identity
        self.transform = Compose(transform) if transform is not None else identity
        self.test_mode = test_mode
        self.CLASSES = self.get_classes(classes)

        self.raw_data_suffix = raw_data_suffix
        if druglib.is_str(raw_data):
            raw_data = [raw_data]
        self.raw_data = raw_data
        self.processed_file_suffix = processed_file_suffix
        if druglib.is_str(processed_file):
            processed_file = [processed_file]
        self.processed_file = processed_file
        if defualt_raw is not None and druglib.is_str(defualt_raw):
            self.defualt_raw = defualt_raw
        if default_processed is not None and druglib.is_str(default_processed):
            self.default_processed = default_processed
        self.kwargs = kwargs

        # inner variable
        self._data_indices: Optional[Sequence] = None
        # execute initial function
        self._init()

        if not test_mode:
            self._set_group_flag()

    def _init(self):
        """Some init executed function module"""
        if self.has_download:
            self._download()
        if self.has_process:
            self._process()

    @property
    def raw_dir(self) -> str:
        """
        Raw data directory;
        E.g.
            1. cv data: img data, this directory will save original image;
            2. graph raw data: any data waiting for graph data to be extracted,
                such ligand (.smi, .sdf, etc) and protein data (.pdb) from PDBBind;
        """
        return osp.join(self.root, self.defualt_raw)

    @property
    def raw_file_name(self) -> Union[List[str], Tuple]:
        """
        The name of the files in the `raw_dir` folder that can skip `download` if there are files.
        If return empty list, then it will download data by :method:`self.download`.
        If args `raw_data` is specified, then directly return the `raw_data`.
        """
        mkdir_or_exists(self.raw_dir)
        return search_dir_files(
            self.raw_dir,
            suffix = self.raw_data_suffix,
            exclude_dir = True,
        ) if self.raw_data is None else self.raw_data

    @property
    def raw_paths(self) -> List[str]:
        """
        The absolute filepaths that must be present in order to skip downloading.
        """
        return [osp.join(self.raw_dir, f) for f in self.raw_file_name]

    @property
    def processed_dir(self) -> str:
        """
        Processed data dir.
        E.g.
            1. cv data: this can be annotations.json, save the img path, size,
                bbox and class label, formatting as
                    [{
                        'filename': 'a.jpg', 'width': 720, 'height': 891,
                        'ann': {
                                'bboxes': np.array (n, 4) in (x1, y1, x2, y2) order,
                                'labels': np.array (n,),
                            }
                    }];
            2. graph data: this can be .pt format file, such as saving the protein graph and
                ligand graph (node, edge, attr feature, etc) and label;
        """
        return osp.join(self.root, self.default_processed)

    @property
    def processed_file_names(self) -> Union[List[str], Tuple]:
        """
        The name of the files in the `processed_dir` folder that can skip `process` if there are files.
        If return empty list, then it will processed raw data by :method:`self.process`.
        If args `processed_file` is specified, then directly return the `processed_file`.
        """
        mkdir_or_exists(self.processed_dir)
        return search_dir_files(
            self.processed_dir,
            suffix = self.processed_file_suffix,
            exclude_dir = True,
        ) if self.processed_file is None else self.processed_file

    @property
    def processed_paths(self) -> List[str]:
        """
        The absolute filepaths that must be present in order to skip processing.
        """
        return [osp.join(self.processed_dir, f) for f in self.processed_file_names]

    @property
    def has_download(self) -> bool:
        """Checks whether the dataset defines a :meth:`download` method."""
        return overrides_method(self.__class__, 'download')

    @property
    def has_process(self) -> bool:
        """Checks whether the dataset defines a :meth:`process` method."""
        return overrides_method(self.__class__, 'process')

    def _download(self):
        if files_exist(self.raw_paths):
            return

        self.download()

    def _process(self):
        if files_exist(self.processed_paths):
            return

        self.logger.info(f'{color.BLUE}Data Processing...{color.END}')
        self.process(**self.kwargs)
        self.logger.info(f'{color.BLUE}Data Processing Done!{color.END}')

    def download(self):
        """Download the dataset to the `self.raw_dir` folder."""
        raise NotImplementedError

    def process(self, *args, **kwargs):
        """
        Process the raw data in `self.raw_dir` folder
            to the `self.processed_dir` folder.
        Args `pre_transform` will be executed in this function.
        """
        self.save_data()

    def save_data(self, *args, **kwargs):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    def format_output(self, data, **kwargs):
        """
        Format processed data to specific output format.
        This is an interface function to formatting the model
            output results to the format required by the evaluation
            server.
        """
        return data

    def __getitem__(
            self, idx: Union[int, np.integer, IndexType],
    ):
        """
        In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices.
        """
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(idx)
            return data
        else:
            return self.index_select(idx)

    def index_select(self, idx: IndexType):
        """
        Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        """
        indices = self.indices

        if isinstance(idx, slice):
            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple = False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        dataset._data_indices = indices
        dataset.flag = dataset.flag[indices]
        return dataset

    def shuffle(
        self,
        return_perm: bool = False,
    ):
        """
        Randomly shuffles the examples in the dataset.
        Args:
            return_perm: bool, optional: If set to :obj:`True`, will also
                return the random permutation used to shuffle the dataset.
                (default: :obj:`False`)
        """
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def get(self, idx):
        """Get one single sample (and label)"""
        if self.test_mode:
            return self._prepare_test_sample(self.indices[idx])
        else:
            seen_count, iter_count = defaultdict(int), 0
            while True and seen_count[idx] < 2 and iter_count < 10:
                seen_count[idx] += 1
                data = self._prepare_train_sample(self.indices[idx])
                if data is None:
                    idx = self._rand_another(idx)
                    iter_count += 1
                    continue
                return data
            raise RuntimeError('Random sampling exceeds limit (10 runs) or duplicated indices.')

    def _pre_pipeline(self, data):
        """Prepare data dict for pipeline"""
        return data

    def _prepare_train_sample(self, idx: int):
        """
        Get training data and labels ofter on-line pipeline.
        The process is as follows:
            1. get a sample (and label) by idx;
            2. save the sample and label in a dict;
            3. if pre-pipline exists, do it;
            4. finally, sequentially execute pipeline starting from the dict;
            5. return training data.
        """
        raise NotImplementedError

    def _prepare_test_sample(self, idx: int):
        """
        Get testing data ofter on-line pipeline.
        The process is as follows:
            1. get a sample by idx;
            2. save the sample in a dict;
            3. if pre-pipline exists, do it;
            4. finally, sequentially execute pipeline starting from the dict;
            5. return testing data.
        """
        raise NotImplementedError

    @classmethod
    def get_classes(cls, classes: Union[str, Sequence[str], None] = None):
        """
        Get classes from current dataset.
        Args:
            classes: str, Sequence[str], Optional. If classes is None,
                use default CLASSES defined by builtin dataset; If classes
                is string, it must be a path file; If classes is Sequence of
                string, such as List of string, then we directly return it.
        Return:
            Sequence[str]: Names of classes of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if druglib.is_str(classes):
            class_name = list_from_file(classes)
        elif isinstance(classes, Sequence):
            class_name = classes
        else:
            raise ValueError(f"Unsupported type {type(classes)} of classes")

        return class_name

    def _rand_another(
            self,
            idx: int,
    ) -> int:
        """Get another random index from the same graph as the given idx"""
        where = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(where)# noqa

    def _set_group_flag(self):
        """
        Set flag, and act as data clustering.
        Default to the same flag.
        This function will be called by GroupSampler.
        """
        self.flag: np.ndarray = np.zeros(len(self), dtype = np.uint8)

    @abstractmethod
    def evaluate(
            self,
            results: List,
            logger: Optional[logging.Logger] = None,
            **kwargs,
    ):
        pass

    def len(self):
        raise NotImplementedError

    @property
    def indices(self) -> Sequence:
        return self._data_indices if self._data_indices is not None else range(self.len())

    def __len__(self):
        return len(self.indices)

    def __repr__(self) -> str:
        len_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({len_repr})'


################################## helper function
def identity(x):
    return x

def files_exist(files: List[str]) -> bool:
    return len(files) != 0 and all([osp.exists(f) for f in files])

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

def overrides_method(
        cls,
        method_name: str,
):
    if method_name in cls.__dict__:
        return True

    out = False
    for base in cls.__bases__:
        if base != CustomDataset:
            out |= overrides_method(base, method_name)
    return out