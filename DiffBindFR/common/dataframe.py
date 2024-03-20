# Copyright (c) MDLDrugLib. All rights reserved.
from typing import (
    Optional,
    Union,
    List,
    Sequence
)
import logging
import argparse
import itertools
from pathlib import Path
from collections import defaultdict
import pandas as pd

from druglib.ops import parse_lig_center
from druglib.utils import get_logger


must_have_cols = [
    'protein',
    'protein_name',
    'ligand',
    'ligand_name',
    'complex_name',
]
any_cols = [
    'crystal_ligand',
    'center',
]

def input_csv(
        csv: Union[str, Path, pd.DataFrame],
) -> pd.DataFrame:
    """
    Check the input csv from users.
        1. Limits the format for versatility
        2. Allow customization
    """
    if not isinstance(csv, pd.DataFrame):
        csv = Path(csv)
        if not csv.exists():
            raise FileNotFoundError(csv)
        df = pd.read_csv(csv)
    else:
        df = csv

    columns = df.columns.tolist()
    must_flag = all(col in columns for col in must_have_cols)
    if not must_flag:
        raise ValueError("columns {} must be specified in dataframe.".format(",".join(must_have_cols)))
    any_flag = any(col in columns for col in any_cols)
    if not any_flag:
        raise ValueError("At least one of columns {} must be specified in dataframe.".format(",".join(any_cols)))

    return df

def single_path(p: Path, suffix: str = 'sdf') -> List[Path]:
    # check input object is directory
    if p.is_dir():
        fs = list(p.glob(f'*.{suffix}'))
    elif p.is_file():
        if not (p.suffix == f'.{suffix}'):
            raise ValueError(f'Only {suffix.upper()} files are supported, but got {p}')
        fs = [p]
    else:
        raise ValueError(f'{p} not found.')
    return fs

def input_object(
        obj: Union[Path, List[Path]],
        suffix: str = 'sdf',
) -> List[Path]:
    """Directory to molecule files or multiple molecules files"""
    if isinstance(obj, str):
        obj = [Path(obj)]
    elif not isinstance(obj, Sequence):
        raise TypeError(f'Input must be path or a list of path, but got {type(obj)}')

    o = []
    for p in obj:
        fs = single_path(Path(p), suffix = suffix)
        o.extend(fs)
    o = list(set(o))
    o = sorted(o)

    return o

def input_ligand(
        ligand: Union[Path, List[Path]],
):
    suffix = 'sdf'
    return input_object(ligand, suffix)

def input_receptor(
        receptor: Union[Path, List[Path]],
) -> List[Path]:
    suffix = 'pdb'
    return input_object(receptor, suffix)

def find_bs_file(
        receptor: List[Path],
):
    """
    Find pocket location definition file.
    1. binding site box csv file (center column)
    2. crystal ligand sdf file (crystal_ligand column)
    """
    ret = {
        'center': [],
        'crystal_ligand': [],
        'all_center_exists': False,
        'all_crystal_exists': True,
    }
    for rec in receptor:
        stem = rec.stem
        crystal_sdf = rec.parent / f'{stem}_crystal.sdf'
        if not crystal_sdf.exists():
            ret['all_crystal_exists'] = False
            break
        ret['crystal_ligand'].append(crystal_sdf)

    if ret['all_crystal_exists']:
        return ret

    for rec in receptor:
        stem = rec.stem
        box_csv = rec.parent / f'{stem}_box.csv'
        crystal_sdf = rec.parent / f'{stem}_crystal.sdf'
        if box_csv.exists():
            with open(str(box_csv), 'r') as fin:
                strings = fin.readlines()[0]
        elif crystal_sdf.exists():
            strings = parse_lig_center(str(crystal_sdf))
        else:
            return ret

        box_str = strings.strip().split(',')
        box_str = [x.strip() for x in box_str]
        ret['center'].append(','.join(box_str[:3]))

    return ret

def assign_id(
        ps: List[Path],
        prefix: str = '',
):
    ids = []
    for idx, p in enumerate(ps):
        stem = p.stem
        if stem in ids:
            stem = f'{prefix}{idx}_{stem}'
        ids.append(stem)

    return ids

def all_against_all(
        ligand: List[Path],
        receptor: List[Path],
) -> pd.DataFrame:
    # define pocket
    ret = find_bs_file(receptor)
    if ret['all_crystal_exists']:
        k = 'crystal_ligand'
    elif ret['all_center_exists']:
        k = 'center'
    else:
        raise ValueError('Find missing crystal ligand sdf and center box csv file configuration')

    # make unique ID fetched from file stem
    rids = assign_id(receptor, 'RID')
    lids = assign_id(ligand, 'LID')
    paired = itertools.product(
        zip(ligand, lids),
        zip(receptor, ret[k], rids)
    )

    # determine object name
    df = defaultdict(list)
    for (l, lid), (r, loc, rid) in paired:
        df['protein'].append(str(r.absolute()))
        df['protein_name'].append(rid)
        df['ligand'].append(str(l.absolute()))
        df['ligand_name'].append(lid)
        df['complex_name'].append(f'{lid}_dok_{rid}')
        df[k].append(loc)

    df = pd.DataFrame(df)
    return df

def make_inference_jobs(
        args: argparse.Namespace,
) -> pd.DataFrame:
    df = args.input_csv
    if df is None:
        ligand = input_ligand(args.ligand)
        receptor = input_receptor(args.receptor)
        df = all_against_all(ligand, receptor)

    df = input_csv(df)
    return df


def JobSlice(
        df: pd.DataFrame,
        args: argparse.Namespace,
        logger: Optional[logging.Logger] = None,
):
    """Support distributed job submission and slurm job array"""
    if logger is None:
        logger = get_logger(name = 'JobSlicer')

    # parse the job slice
    start = 0
    if args.start is not None:
        start = args.start
    args.start = start

    if args.interval is not None and args.end is None:
        args.end = args.start + args.interval

    df_len = df.shape[0]
    if args.end is None:
        args.end = df_len
    if args.end > df_len:
        args.end = df_len
    assert args.start + 1 < args.end, f'start: {args.start} v.s. end: {args.end}'

    logger.info(f'Total loaded jobs: {df_len}.')
    logger.info(f'Job Slice Info: ({args.start}, {args.end}).')
    df = df.iloc[slice(args.start, args.end)]
    logger.info(f'Running jobs: {df.shape[0]}.')
    df.reset_index(drop=True, inplace=True)

    return df

if __name__ == '__main__':
    from DiffBindFR.common.args import parse_args
    parser = parse_args()
    args = parser.parse_args()
    df = make_inference_jobs(args)
    df.to_csv(args.export_dir, index = False)