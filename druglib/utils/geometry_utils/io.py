# Copyright (c) MDLDrugLib. All rights reserved.
import os
import pickle
import lmdb
from typing import Union, List, Sequence
from pathlib import Path
import numpy as np


def _load(path: Union[Path, str]):
    path = str(path)
    return np.load(path)


def _save(path: Union[Path, str], obj):
    path = str(path)
    return np.save(path, obj)


def _load_lmdb(
        path: Union[Path, str]
):
    env = lmdb.open(
        str(path),
        map_size=int(1e12),
        create=False,
        subdir=os.path.isdir(str(path)),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )

    return env


def _save_lmdb(
        output_file: Union[Path, str],
        data_dict: dict,
):
    env = lmdb.open(
        str(output_file),
        map_size=int(1e12),
        create=True,
        subdir=False,
        readonly=False,
    )
    txn = env.begin(write=True)
    for k, d in data_dict.items():
        txn.put(str(k).encode('ascii'), pickle.dumps(d))
    txn.commit()
    env.close()

    return


def _load_lmdb_data(
        env: lmdb.Environment,
        keys: Union[str, List[str]],
):
    if isinstance(keys, str) or not isinstance(keys, Sequence):
        keys = [keys]

    with env.begin() as txn:
        KEYS = [k for k in txn.cursor().iternext(values=False)]

    ds = []
    for key in keys:
        key = str(key).encode("ascii")
        if key not in KEYS:
            raise ValueError(f'query index {key.decode()} not in lmdb.')

        with env.begin() as txn:
            with txn.cursor() as curs:
                d = pickle.loads(curs.get(key))
                if len(keys) == 1:
                    return d

                ds.append(d)

    return ds