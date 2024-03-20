# Copyright (c) MDLDrugLib. All rights reserved.
import os
import glob
import logging
import pickle
import lmdb
from typing import *
import os.path as osp
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()
from functools import partial
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from .lmdbdataset import LMDBLoader
from .pipeline import single_process


class InferenceScoringDataset(Dataset):

    def __init__(
            self,
            pair_frame: pd.DataFrame,  # protein ligand pair dataframe
            debug: bool = False,
            pocket_radius: float = 12.0,
            save_path: Optional[Union[str, Path]] = None,
            lmdb_mode: bool = False,
            n_jobs: int = 1,
            verbose: bool = False,
            **kwargs,
    ):
        assert isinstance(pair_frame, pd.DataFrame) or pair_frame is None
        self.lmdb_mode = lmdb_mode
        self.debug = debug
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            filename=None,
            datefmt='%Y/%m/%d %H:%M:%S',
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        self.logger = logging.getLogger(name=self.__class__.__name__)

        self.pocket_radius = pocket_radius
        self.pair_frame = pair_frame
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.save_path = save_path
        self.process(**kwargs)

    def process(self, *args, **kwargs):
        self.pair_frame['_index'] = list(range(len(self)))

        if self.lmdb_mode and Path(self.save_path).exists():
            self.logger.info(f'Load precomputed lmdb data from {self.save_path}...')
            self.datas = LMDBLoader(self.save_path)
            self.logger.info('Reload lmdb data is Done!')
            return

        if self.n_jobs <= 1:
            self.logger.info('Seriel processing...')
            if not self.verbose:
                datas = self.pair_frame.apply(lambda x: single_process(x, self.pocket_radius), axis = 1, result_type='expand')
            else:
                datas = self.pair_frame.progress_apply(lambda x: single_process(x, self.pocket_radius), axis=1, result_type='expand')
        else:
            self.logger.info('Multiprocessing...')
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=self.n_jobs, progress_bar=self.verbose)
            datas = self.pair_frame.parallel_apply(lambda x: single_process(x, self.pocket_radius), axis=1)

        datas = datas.tolist()
        if self.lmdb_mode:
            self.logger.info(f'Save processed data to LMDB to path {self.save_path}...')
            datas = self.save_lmdb(datas)

        self.logger.info('Processing is Done!')
        self.datas = datas

    def save_lmdb(self, datas: List[Any]):
        self.logger.info(f'Save {len(datas)} datapoints to LMDB...')
        Path(self.save_path).parent.mkdir(exist_ok=True, parents=True)
        env = lmdb.open(
            self.save_path,
            map_size=int(1e12),
            create=True,
            subdir=False,
            readonly=False,
        )
        txn = env.begin(write=True)
        for _idx, d in enumerate(datas):
            txn.put(str(_idx).encode('ascii'), pickle.dumps(d[0])) # unique: fetch data from tuple
            if _idx % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        env.close()

        del datas
        datas = LMDBLoader(self.save_path)

        return datas

    def __getitem__(self, idx):
        return self.datas[idx]

    def __len__(self):
        return self.pair_frame.shape[0]

class InferenceScoringDataset_chunk(InferenceScoringDataset):
    def process(self, chunk = 50000, suffix = ''):
        suffix = suffix.replace('_', '')
        self.pair_frame['_index'] = list(range(len(self)))

        if self.lmdb_mode and Path(self.save_path).exists():
            self.logger.info(f'Load precomputed lmdb data from {self.save_path}...')
            self.datas = LMDBLoader(self.save_path)
            self.logger.info('Reload lmdb data is Done!')
            return

        data_frame = self.pair_frame
        self.processed_dir = str(Path(self.save_path).parent)
        num_processed = 0
        if chunk > 0:
            tempf = glob.glob(osp.join(self.processed_dir, f"*temp{suffix}_*.pt"))
            if len(tempf) > 0:
                num_processed = sum([len(torch.load(tf)) for tf in tempf])

        assert isinstance(data_frame, pd.DataFrame), \
            'When processing data, pair frame should be specified to pd.DataFrame.'
        self.logger.debug(f'Previous Processed number: {num_processed}')
        num_jobs = data_frame.iloc[num_processed:].shape[0]
        if chunk < 0: chunk = num_jobs
        num_iters = int(np.ceil(num_jobs / chunk))
        Iter = tqdm(
            range(num_iters), desc=f'Processing {num_jobs} jobs, '
            f'chunk {chunk} per iter, total {num_iters} iters...') if self.debug else \
            range(num_iters)

        if self.n_jobs <= 1:
            self.logger.info('Seriel processing...')
        else:
            self.logger.info('Multiprocessing...')
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=self.n_jobs, progress_bar=self.verbose)

        process_fn = partial(single_process, pocket_radius=self.pocket_radius)
        data_list = []
        for _ in Iter:
            start = num_processed
            end = num_processed + chunk
            if self.n_jobs > 1:
                data_list = data_frame.iloc[start:end].parallel_apply(process_fn, axis=1)
            else:
                if not self.verbose:
                    data_list = data_frame.iloc[start:end].apply(
                        process_fn, axis=1)
                else:
                    data_list = data_frame.iloc[start:end].progress_apply(
                        process_fn, axis=1)
            data_list = data_list.tolist()
            num_processed += len(data_list)
            tempid = int(np.ceil(num_processed / chunk))
            if chunk > 0:
                tempf = osp.join(
                    self.processed_dir,
                    f'temp{suffix}_{tempid}.pt'
                )
                torch.save(data_list, tempf)
                self.logger.info(
                    f'Dump {tempid}th data batch with length {len(data_list)} '
                    f'to {tempf}. Clean up the data list...')

        if chunk > 0:
            tempf = glob.glob(osp.join(self.processed_dir, f"*temp{suffix}_*.pt"))
            tempf = sorted(
                tempf, reverse=True,
                key=lambda n: n.split(f'temp{suffix}_')[1].split('.')[0]
            )
            data_list = []
            for tf in tempf:
                data_list = torch.load(tf) + data_list
            for f in tempf: os.remove(f)

        assert len(data_list) == data_frame.shape[0], \
            f'Generated data number ({len(data_list)}) mismatches data frame {data_frame.shape[0]}'

        if self.lmdb_mode:
            self.logger.info(f'Save processed data to LMDB to path {self.save_path}...')
            data_list = self.save_lmdb(data_list)

        self.logger.info('Processing is Done!')
        self.datas = data_list