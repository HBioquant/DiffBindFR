# Copyright (c) MDLDrugLib. All rights reserved.
import os, random, glob, logging, copy, gc, pickle
from typing import Optional, Union, Tuple, List, Any, Callable
import os.path as osp
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict, namedtuple
import pandas as pd
import numpy as np
import lmdb
import torch
from torch import Tensor

import druglib
from druglib.utils.obj import Protein, Ligand3D
from druglib.datasets import CustomDataset, LMDBLoader
from druglib.datasets.base_pipelines import Compose
from druglib.datasets.Docking.utils import nx_from_Ligand3D


InputProteinMeta = namedtuple(
    'InputProteinMeta',
    [
        'protein',
        'pocket',
        'pocket_mask',
        'atom14_position',
        'atom14_mask',
        'torsion_angle',
        'sc_torsion_edge_mask',
        'pocket_center_pos', # input data has been moved to origin, output structure should be moved back
        'model_input',
        'lig_randomized_center',
    ],
    defaults = (None, ) * 7
)
InputLigandMeta = namedtuple(
    'InputLigandMeta',
    [
        'ligand', # including lig_pos (np.ndarray)
        'num_nodes',
        'num_edges',
        'match_nx', # For heavy atom symmetry-corrected RMSD
        'model_input',
    ],
    defaults = (None, ) * 4
)

Traj = namedtuple(
    'Traj',
    [
        'ligand',
        'protein',
    ],
)

def add_center_pos(
        pos: Tensor,
        center: Tensor,
) -> Tensor:
    shape = pos.shape
    pos = pos + center.view(*((1,) * len(shape[:-1]) + (3,)))
    return pos

# EValuation and Inference Data Container
class PLPairDynData(object):
    def __init__(
            self,
            cfg = None,
    ):
        self.cfg = cfg if cfg is not None else {}
        self.clean()

    def clean(self) -> None:
        # define data save group
        self.proteins: Union[dict, LMDBLoader] = dict()
        self.ligands: Union[dict, LMDBLoader] = dict()
        self.traj_group = OrderedDict()

    def put_protein(
        self,
        name: Any,
        metadata: dict,
        model_input_data: Optional[dict] = None,
    ) -> None:
        if not isinstance(metadata, dict):
            raise TypeError(f'Expect dict, but got {type(metadata)}')

        proteininfo = InputProteinMeta(
            protein = metadata.pop('protein'),
            pocket = metadata.pop('pocket'),
            pocket_mask = metadata.pop('pocket_mask'),
            atom14_position = metadata['atom14_position'],
            # save mask because missing side chain will be fixed
            atom14_mask = metadata['atom14_mask'],
            torsion_angle = metadata['torsion_angle'],
            sc_torsion_edge_mask = metadata['sc_torsion_edge_mask'],
            pocket_center_pos = metadata.get('pocket_center_pos'),
            model_input = model_input_data,
            lig_randomized_center = metadata.pop('pocket_sel_center', None),
        )
        self.proteins[name] = proteininfo

    def put_ligand(
            self,
            name: Any,
            metadata: dict,
            model_input_data: Optional[dict] = None,
            use_nx: bool = True,
            add_edge_match: bool = True,
    ) -> None:
        if not isinstance(metadata, dict):
            raise TypeError(f'Expect dict, but got {type(metadata)}')

        metastore = metadata['metastore']
        match_nx = nx_from_Ligand3D(metadata['ligand'], add_edge_match) if use_nx else None
        ligandinfo = InputLigandMeta(
            ligand = metadata.pop('ligand'),
            num_nodes = metastore.pop('num_nodes'),
            num_edges = metastore.pop('num_edges'),
            match_nx = match_nx,
            model_input = model_input_data,
        )
        self.ligands[name] = ligandinfo

    def put_inference(
            self,
            name: Any,
            ligand_traj: Tensor,
            protein_traj: Tensor,
    ) -> None:
        traj = Traj(
            ligand = ligand_traj,
            protein = protein_traj,
        )
        self.traj_group[name] = traj

    def moveback(
            self,
            traj_name: Any,
            pocket_center_pos: Union[Tensor, np.ndarray],
            inplace: bool = True,
    ):
        """Move the denoised trajetories back to original system coordinates"""
        assert traj_name in self.traj_group
        traj: Traj = self.traj_group[traj_name]
        ligand_traj = traj.ligand
        protein_traj = traj.protein
        if isinstance(pocket_center_pos, np.ndarray):
            pocket_center_pos = torch.from_numpy(pocket_center_pos)
        pocket_center_pos = pocket_center_pos.to(ligand_traj.device)
        ligand_traj = add_center_pos(ligand_traj, pocket_center_pos)
        protein_traj = add_center_pos(protein_traj, pocket_center_pos)
        traj = Traj(
            ligand = ligand_traj,
            protein = protein_traj,
        )
        if inplace: self.traj_group[traj_name] = traj

        return traj


class InferenceDataset(CustomDataset):
    def __init__(
            self,
            root: Union[str, Path],
            cfg: druglib.ConfigDict,
            pair_frame: pd.DataFrame, # protein ligand pair dataframe
            num_poses: int = 10,
            init_from_cry_lig: bool = False,
            generate_multiple_conformer: bool = False,
            default_processed: Optional[str] = None,
            batch_repeat: bool = False,
            debug: bool = False,
            lmdb_mode: bool = False,
            **kwargs
    ):
        assert isinstance(pair_frame, pd.DataFrame) or pair_frame is None
        self.cfg = cfg
        self.lmdb_mode = lmdb_mode
        self.debug = debug
        logging.basicConfig(
            level = logging.DEBUG if debug else logging.INFO,
            filename = None,
            datefmt = '%Y/%m/%d %H:%M:%S',
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        self.logger = logging.getLogger(name = self.__class__.__name__)

        self.lig_pre_transform = Compose(self.cfg.test_pre_transform_lig.copy())
        self.prot_pre_transform = Compose(self.cfg.test_pre_transform_prot.copy())
        self.crystal_loader = Compose(self.cfg.cystal_ligand_loader.copy())
        self.formatter = Compose(self.cfg.collect_transform.copy())

        self.init_from_cry_lig = init_from_cry_lig
        if init_from_cry_lig:
            self.logger.info('Randomized initialization of ligand binding pose from crystal ligand center.')

        self.batch_repeat = batch_repeat
        self.num_poses = num_poses
        self.logger.info(
            f'Make sequential identical mols in batch, and generate {num_poses} poses.' if batch_repeat else
            f'Make heterogeneous mols in batch, and generate {num_poses} poses.'
        )
        self.generate_multiple_conformer = generate_multiple_conformer
        if generate_multiple_conformer:
            self.logger.info(f'Generate rdkit multiple conformers for initialization.')

        self.pair_frame = pair_frame
        self.PairDataInit()

        super().__init__(
            root = str(root),
            test_mode = True,
            transform = self.cfg.test_transform.copy(),
            default_processed = default_processed,
            **kwargs
        )

        self.load_data()

    def PairDataInit(self):
        """
        PairData Plugin, this allow the dataset to extend
            the pair data type storage.
        """
        self.PairData = PLPairDynData({})

    @property
    def protein_index(self):
        return 'protein_name' if 'protein_name' in self.pair_frame.columns else 'protein'

    @property
    def ligand_index(self):
        return 'ligand_name' if 'ligand_name' in self.pair_frame.columns else 'ligand'

    @property
    def pocket_index(self):
        return 'center' if 'center' in self.pair_frame.columns else 'crystal_ligand'

    @property
    def complex_index(self) -> Optional[str]:
        """
        Make data iterative pair name.
        Here we support keyword `complex_name` or `pair_name`
            _pair_name means made from protein(_name) or ligand(_name)
        """
        for x in ['complex_name', 'pair_name', '_pair_name']:
            if x in self.pair_frame.columns:
                return x
        return None

    @property
    def processed_file_names(self) -> Union[List[str], Tuple]:
        druglib.mkdir_or_exists(self.processed_dir)
        if self.lmdb_mode:
            return ['PairFrame.pt', 'proteins.lmdb', 'ligands.lmdb']
        else:
            return ['PairFrame.pt', 'proteins.pt', 'ligands.pt']

    def process(
            self,
            n_jobs: int = 1,
            verbose: bool = False,
            **kwargs,
    ):
        assert isinstance(self.pair_frame, pd.DataFrame), \
            'When processing data, pair frame should be specified to pd.DataFrame.'
        # pre-processing
        for x in ['protein', 'ligand']:
            self.pair_frame[x] = self.pair_frame[x].apply(lambda x: osp.abspath(x))
        self.make_indices()

        def unique_columns(df, columns, sort = False):
            if isinstance(columns, str):
                columns = [columns]
            all_cols = df.columns
            df = df.groupby(columns, sort = sort).agg(
                {k: 'first' for k in all_cols if k not in columns}).reset_index()
            return df

        prot_frame = unique_columns(
            self.pair_frame,
            ['protein', 'center'] if 'center' in self.pair_frame.columns else
            ['protein', 'crystal_ligand'],
        )
        lig_frame = unique_columns(self.pair_frame, 'ligand')

        def lig_processer(row):
            try:
                lig_data = self.lig_pre_transform(
                    dict(
                        ligand_file = row['ligand'],
                    )
                )
            except Exception as e:
                if not self.debug:
                    self.logger.error(f'{str(e)} from {row["ligand"]}')
                else:
                    raise Exception(f"{row['ligand']}: {str(e)}")
                lig_data = None

            return lig_data

        def prot_processer(row):
            
            try:
                # use crystal ligand geometric center to get pocket
                if 'center' in row:
                    pocket_sel_center = row['center']
                    pocket_sel_center = [float(x) for x in pocket_sel_center.split(',')]
                    assert len(pocket_sel_center), f'Point3d (x, y, z), but got {pocket_sel_center}'
                    process_input = dict(
                        protein_file = row['protein'],
                        pocket_sel_center = pocket_sel_center,
                    )
                elif 'crystal_ligand' in row:
                    cry_lig_data = self.crystal_loader(
                        dict(
                            ligand_file = row['crystal_ligand'],
                        )
                    )
                    cry_lig_data.pop('ligand_file', None)
                    process_input = dict(
                        protein_file=row['protein'],
                        ligand=cry_lig_data.pop('ligand'),
                    )
                else:
                    raise NotImplementedError('Pocket should be defined by ligand (or its 3d center)')

                pocket_data = self.prot_pre_transform(process_input)
            except Exception as e:
                if not self.debug:
                    if 'center' in row:
                        self.logger.error(f'{str(e)} from {row["protein"]} at center {row["center"]}')
                    elif 'crystal_ligand' in row:
                        self.logger.error(f'{str(e)} from {row["protein"]} from crystal ligand {row["crystal_ligand"]}')
                else:
                    raise Exception(f"{row['protein']}: {str(e)}")
                pocket_data = None

            return pocket_data

        if n_jobs > 1:
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers = n_jobs, progress_bar = verbose)

        lig_datas, lig_failed_ids = self.data_processing(
            lig_frame,
            process_fn = lig_processer,
            column_name = 'ligand',
            n_jobs = n_jobs,
            **kwargs
        )
        prot_datas, prot_failed_ids = self.data_processing(
            prot_frame,
            process_fn = prot_processer,
            column_name = 'protein',
            n_jobs = n_jobs,
            **kwargs,
        )
        failed_ids = list(set(lig_failed_ids + prot_failed_ids))

        if len(failed_ids) > 0:
            failed_frame = self.pair_frame.iloc[failed_ids]
            self.logger.info(f'Save failed csv to directory {self.processed_dir} and update the input pair frame...')
            failed_frame.to_csv(osp.join(self.processed_dir, 'failed.csv'), index = False)
            self.pair_frame.drop(failed_ids, inplace = True)
            self.make_indices()

        for fnm, d in prot_datas:
            if fnm not in self.PairData.proteins:
                self.PairData.put_protein(
                    name = fnm, metadata = d,
                    model_input_data = d,
                )
                prot: Protein = self.PairData.proteins[fnm].protein
                for _d in [prot.atom_prop, prot.residue_prop]:
                    if _d is not None:
                        for k in list(_d.keys()): _d.pop(k)

        for fnm, d in lig_datas:
            if fnm not in self.PairData.ligands:
                self.PairData.put_ligand(
                    name = fnm, metadata = d,
                    model_input_data = d,
                )
                lig: Ligand3D = self.PairData.ligands[fnm].ligand
                for _d in [lig.atom_prop, lig.bond_prop]:
                    if _d is not None:
                        for k in list(_d.keys()): _d.pop(k)

        self.save_data()

    def data_processing(
            self,
            data_frame: pd.DataFrame,
            process_fn: Callable,
            column_name: str = '',
            n_jobs: int = 1,
            chunk: int = 1000,
            remove_temp: bool = False,
    ):
        num_processed = 0
        if chunk > 0:
            tempf = glob.glob(osp.join(self.processed_dir, f"*temp_{column_name}_*.pt"))
            if len(tempf) > 0:
                num_processed = sum([len(druglib.load(tf)) for tf in tempf])

        assert isinstance(data_frame, pd.DataFrame), \
            'When processing data, pair frame should be specified to pd.DataFrame.'
        self.logger.debug(f'Previous Processed {column_name} number: {num_processed}')
        num_jobs = data_frame.iloc[num_processed:].shape[0]
        if chunk < 0:
            chunk = num_jobs
        num_iters = int(np.ceil(num_jobs / chunk))
        Iter = tqdm(range(num_iters), desc = f'Processing {num_jobs} {column_name} jobs, '
                    f'chunk {chunk} per iter, total {num_iters} iters...') if self.debug else range(num_iters)

        data_list = []
        for _ in Iter:
            start = num_processed
            end = num_processed + chunk
            if n_jobs > 1:
                data_list = data_frame.iloc[start:end].parallel_apply(process_fn, axis=1)
                data_list = data_list.tolist()
            else:
                data_list = []
                for _, row in data_frame.iloc[start:end].iterrows():
                    data_list.append(process_fn(row))

            num_processed += len(data_list)
            tempid = int(np.ceil(num_processed / chunk))
            if chunk > 0:
                tempf = osp.join(
                    self.processed_dir,
                    f'temp_{column_name}_{tempid}.pt'
                )
                druglib.dump(data_list, tempf)
                self.logger.info(
                    f'Dump {tempid}th data batch with length {len(data_list)} '
                    f'to {tempf}. Clean up the data list...')

        if chunk > 0:
            tempf = glob.glob(osp.join(self.processed_dir, f"*temp_{column_name}_*.pt"))
            tempf = sorted(
                tempf, reverse = True,
                key = lambda n: n.split(f'temp_{column_name}_')[1].split('.')[0]
            )
            data_list = []
            for tf in tempf:
                data_list = druglib.load(tf) + data_list

            if remove_temp:
                for f in tempf: os.remove(f)

        assert len(data_list) == data_frame.shape[0], \
            f'Generated data number ({len(data_list)}) mismatches data frame {data_frame.shape[0]} at column "{column_name}"'

        # collect failed cases and write the failed to processed diretory for further processing
        failed_id, _data_list = [], []
        name = getattr(self, f'{column_name}_index')
        for rowid, d in enumerate(data_list):
            if d is None:
                # use index id to query failed row id in pair frame
                # because we do not know what type the column name is
                # but we have initialized the index in the pair frame
                index = data_frame.iloc[rowid]['index']
                query_frame = self.pair_frame.query(f'index == {index}')
                failed_id.extend(query_frame['index'].tolist())
            else: _data_list.append((data_frame.iloc[rowid][name], d))
        data_list = _data_list

        self.logger.info(f'Final report of column {column_name} precessing: processed {num_processed}, '
                         f'and got {num_processed - len(failed_id)} successful datas')

        # return the failed row id in `column_name`
        return data_list, failed_id

    def make_indices(self):
        self.pair_frame['index'] = list(range(self.pair_frame.shape[0]))

        def _repeat(ls):
            from itertools import chain, repeat
            repeat_iter = repeat(ls, self.num_poses)
            if self.batch_repeat: repeat_iter = zip(*repeat_iter)
            return list(chain.from_iterable(repeat_iter))

        # repeat dataset by using indices
        self._data_indices = _repeat(range(self.pair_frame.shape[0]))

        # make pair name if users do not provide keywords
        if self.complex_index is None:
            pair_names = list(zip(self.pair_frame[self.protein_index].tolist(),
                                  self.pair_frame[self.ligand_index].tolist()))
            pair_names = ['>'.join(x) for x in pair_names]
            self.pair_frame['_pair_name'] = pair_names
        else:
            pair_names = self.pair_frame[self.complex_index].tolist()

        # check duplicates
        if len(np.unique(pair_names)) < len(pair_names): # noqa
            raise ValueError(f'pair name should be unique per row.')

        self.repeat_pair_names = _repeat(pair_names)

    def save_data(self):
        ligands  = self.PairData.ligands
        proteins = self.PairData.proteins
        pdir     = self.processed_dir

        druglib.dump(self.pair_frame, osp.join(pdir, self.processed_file_names[0]))

        if self.lmdb_mode:
            self.logger.info(f'Save proteins and ligands LMDB data to {pdir}.')
            for idx, db in enumerate([proteins, ligands], start = 1):
                env = lmdb.open(
                    osp.join(pdir, self.processed_file_names[idx]),
                    map_size = int(1e12),
                    create = True,
                    subdir = False,
                    readonly = False,
                )
                txn = env.begin(write=True)
                for _idx, (k, d) in enumerate(db.items()):
                    txn.put(str(k).encode('ascii'), pickle.dumps(d))
                    if _idx % 1000 == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                txn.commit()
                env.close()

            # reload data to reduce the memory
            del self.PairData
            self.load_data()
        else:
            self.logger.info(f'Save proteins and ligands .pt data to {pdir}.')
            druglib.dump(
                proteins, osp.join(pdir, self.processed_file_names[1])
            )
            druglib.dump(
                ligands, osp.join(pdir, self.processed_file_names[2])
            )
        self.logger.info('Save data Done!')
        gc.collect()

    def load_data(self):
        self.pair_frame = druglib.load(osp.join(self.processed_dir, self.processed_file_names[0]))
        self.PairDataInit()

        load_fn = druglib.load
        if self.lmdb_mode:
            load_fn = LMDBLoader

        self.PairData.proteins = load_fn(
            osp.join(self.processed_dir, self.processed_file_names[1])
        )
        self.PairData.ligands  = load_fn(
            osp.join(self.processed_dir, self.processed_file_names[2])
        )

        self.make_indices()

    def index(self, idx: Union[int, List[int], slice]):
        """
        This provides the dataset.index(1)[key] functionality.
        Return the indexed pair frame.
        """
        return self.pair_frame.iloc[idx]

    def index_complex(self, complex_name: str):
        """Return the ligand index or protein index of the complex for PairData indexing"""
        frame = self.pair_frame.query(f'{self.complex_index} == "{complex_name}"')
        protein_name = frame[self.protein_index]
        ligand_name = frame[self.ligand_index]
        return protein_name, ligand_name

    def _prepare_test_sample(self, idx: int):
        row = self.pair_frame.iloc[idx]
        model_input = dict()
        for obj in ['protein', 'ligand']:
            name = getattr(self, f'{obj}_index')
            mdl_inp = getattr(self.PairData, obj + 's')[row[name]].model_input
            assert mdl_inp is not None, f'model input from {row[obj]} is None'
            model_input.update(copy.deepcopy(mdl_inp))

        # DiffBindFR cannot adjust the conformer of ring, so we need to
        # prepare multiple rdkit conformations just like conventional docking (Dock3.7)
        if self.generate_multiple_conformer:
            mol_obj = self.PairData.ligands[row[self.ligand_index]].ligand.model
            CIDs = [x.GetId() for x in mol_obj.GetConformers()]
            CID  = random.choice(CIDs)
            model_input['lig_pos'] = torch.from_numpy(
                mol_obj.GetConformer(CID).GetPositions()
            ).float()

        model_input = self.formatter(model_input)
        model_input = self.transform(model_input)

        if self.init_from_cry_lig:
            pmdl_inp = self.PairData.proteins[row[self.protein_index]]
            pocket_center_pos = pmdl_inp.pocket_center_pos
            lig_randomized_center = pmdl_inp.lig_randomized_center
            # move docked ligands to crystal ligand center for randomized initialization
            model_input['lig_pos'].update_(
                model_input['lig_pos'].data + (
                        lig_randomized_center.reshape(1, 3) -
                        pocket_center_pos.reshape(1, 3)
                )
            )

        return model_input

    def len(self):
        return len(self._data_indices)

    def evaluate(self, **kwargs):
        raise NotImplementedError

    def _prepare_train_sample(self, idx: int):
        raise NotImplementedError(f'This dataset :cls:`{self.__class__.__name__}` is only used to inference')

    def __repr__(self) -> str:
        len_repr = str(self.pair_frame.shape[0]) + f'x{self.num_poses}'
        return f'{self.__class__.__name__}({len_repr})'
