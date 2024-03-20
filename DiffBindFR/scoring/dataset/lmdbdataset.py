# Copyright (c) MDLDrugLib. All rights reserved.
import os, pickle
from typing import *
import lmdb
from functools import lru_cache


class LMDBLoader:
    def __init__(
            self,
            db_path: str,
            map_gb: float = 10000.0,
            strict_get: bool = True,
            _exclude_key: List[str] = ['KEYS'],
    ):
        if not os.path.exists(db_path):
            raise ValueError("{} does not exists.".format(db_path))

        self.db_path = db_path
        self.map_gb = map_gb
        self.strict_get = strict_get
        self._exclude_key = _exclude_key

        env = self._connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = [k for k in txn.cursor().iternext(values=False) if k.decode() not in _exclude_key]

        import atexit
        atexit.register(lambda s: s._close_db, self)

    def _connect_db(
            self,
            lmdb_path: str,
            attach: bool = False,
    ) -> Optional[lmdb.Environment]:
        assert getattr(self, '_env', None) is None, 'A connection has already been opened.'
        env = lmdb.open(
            lmdb_path,
            map_size=int(self.map_gb * (1024 * 1024 * 1024)),
            create=False,
            subdir=os.path.isdir(lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not attach:
            return env
        else:
            self._env = env

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: str):
        return key.encode("ascii") in self._keys

    @lru_cache(maxsize=16)
    def __getitem__(self, idx) -> Optional[Any]:
        if not hasattr(self, "_env"):
            self._connect_db(self.db_path, attach = True)

        idx = str(idx).encode("ascii")
        if idx not in self._keys:
            if self.strict_get:
                raise ValueError(f'query index {idx.decode()} not in lmdb.')
            else:
                return None

        with self._env.begin() as txn:
            with txn.cursor() as curs:
                datapoint_pickled = curs.get(idx)
                data = pickle.loads(datapoint_pickled)

        return data

    def _close_db(self):
        if hasattr(self, '_env') and \
                isinstance(self._env, lmdb.Environment):
            self._env.close()
            self._env = None