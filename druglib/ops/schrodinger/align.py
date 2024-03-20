# Copyright (c) MDLDrugLib. All rights reserved.
import os, logging, shutil, subprocess
import os.path as osp
from pathlib import Path


logging.basicConfig(
    level = logging.INFO,
    filename = None,
    datefmt = '%Y/%m/%d %H:%M:%S',
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(
    name = 'SchAlign',
)

def must_be_type(
        var,
        _type,
):
    if not isinstance(var, _type):
        raise TypeError(f'Expected type {_type}, but got type {type(var)}')
    return var

def must_be_bool(
        var,
):
    return must_be_type(var, bool)

def must_be_str(
        var,
):
    return must_be_type(var, str)

def parse_rmsd(csv_file):
    if not osp.exists(csv_file):
        logger.warning(f'{csv_file} do not exists.')
        return (None, None)

    with open(csv_file, 'r') as fin:
        rmsds = fin.readlines()
    if len(rmsds) < 2:
        logger.warning(f'{csv_file} has no rmsd records.')
        return (None, None)

    rmsds = rmsds[1].split(',')
    if len(rmsds) != 4:
        logger.warning(f'{csv_file} record must be three-commas-separated.')
        return (None, None)

    rmsds = rmsds[2:]
    if len(rmsds) != 2:
        logger.warning(f'rmsds should be before RMSD and after RMSD, but got number {len(rmsds)} from {csv_file}.')
        return (None, None)

    rmsds = [round(float(rmsd.strip()), 3) for rmsd in rmsds]
    return rmsds

def bs_algn(
        target_file: str,
        mobile_file: str,
        output_file: str,
        exec_bin: str,
        remove_temp: bool = False,
        **kwargs,
):
    # parse arguments
    cutoff = kwargs.get('cutoff', 5)
    dist = kwargs.get('dist', 5)
    lignum = ''
    residues = ''
    jobname = ''  # default: input file basename
    rmsd_type = ''  # Calpha
    host = ''
    prealigned = must_be_bool(kwargs.get('prealigned', False))
    inplace = must_be_bool(kwargs.get('inplace', False))
    show_color = must_be_bool(kwargs.get('show_color', False))
    wait = must_be_bool(kwargs.get('wait', True))

    if 'lignum' in kwargs:
        lignum = '-l ' + kwargs['lignum']

    if 'residues' in kwargs and len(kwargs['residues']) > 0:
        residues = kwargs['residues'].split(',')
        assert len(residues) > 2, 'At least three residues are required and comma-separated strings'
        residues = ','.join([r.strip() for r in residues])  # comma-separated list should not have spaces
        residues = '-r ' + residues

    if 'jobname' in kwargs:
        jobname = '-j ' + kwargs['kwargs']

    if 'rmsd_type' in kwargs:
        rmsd_type = '-rmsdasl ' + kwargs['rmsd_type']

    if 'host' in kwargs:
        host = '-HOST ' + kwargs['host']

    if prealigned:
        prealigned = '-p'
    else:
        prealigned = ''

    if inplace:
        inplace = '-inplace'
    else:
        inplace = ''

    if show_color:
        show_color = '-color'
    else:
        show_color = ''

    if wait:
        wait = '-WAIT'
    else:
        wait = ''

    output_dir = osp.dirname(osp.abspath(output_file))
    output_fn = osp.basename(output_file)
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(target_file, osp.join(output_dir, '_target.pdb'))
    shutil.copy(mobile_file, osp.join(output_dir, '_mobile.pdb'))

    output_str = '-o ' + output_fn
    command = f'cd {output_dir} && '
    algn_cmd = ' '.join([str(p) for p in [exec_bin, lignum, f'-c {cutoff}', f'-d {dist}', prealigned, inplace, residues,
                                          jobname, show_color, rmsd_type, wait, host, output_str, '_target.pdb',
                                          '_mobile.pdb'] if p])
    command = command + algn_cmd

    subprocess.run(
        command, shell = True,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    )

    if remove_temp:
        for tmp in ['_mobile.pdb', '_target.pdb']:
            os.remove(osp.join(output_dir, tmp))

    if not kwargs.get('return_rmsd', False): return (None, None)

    # check whether the program execution normally
    if not osp.exists(output_file):
        logger.warning(f'mobile file {mobile_file} align target {target_file} to {output_file} failed!')
        return (None, None)

    csv_file = osp.join(output_dir, '_target.csv')

    return parse_rmsd(csv_file)