# Copyright (c) MDLDrugLib. All rights reserved.
import argparse, time
from typing import Optional, Sequence
from pathlib import Path
import multiprocessing as mp
import DiffBindFR


def base64_encode(strings: str):
    import base64
    return base64.b64encode(strings.encode('utf-8')).decode('utf-8')

def base64_decode(strings: str):
    import base64
    return base64.b64decode(strings.encode('utf-8')).decode('utf-8')

def _path(path_str: Optional[str]):
    if path_str is None:
        return path_str

    path = Path(path_str).absolute()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File {path} not found!")
    return path

LOGO = ('IF9fX18gIF8gIF9fICBfXyBfX19fICBfICAgICAgICAgICBfIF9fX19fIF9fX18gIAp8ICBfIFwoXykvIF98LyBffCBfX'
        'yApKF8pXyBfXyAgIF9ffCB8ICBfX198ICBfIFwgCnwgfCB8IHwgfCB8X3wgfF98ICBfIFx8IHwgJ18gXCAvIF9gIHwgfF8'
        'gIHwgfF8pIHwKfCB8X3wgfCB8ICBffCAgX3wgfF8pIHwgfCB8IHwgfCAoX3wgfCAgX3wgfCAgXyA8IAp8X19fXy98X3xff'
        'CB8X3wgfF9fX18vfF98X3wgfF98XF9fLF98X3wgICB8X3wgXF9cIA==')


def parse_args():
    print(base64_decode(LOGO), flush = True)
    parser = argparse.ArgumentParser(
        description = 'Welcome to use DiffBindFR (a reliable flexible docking software)',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        add_help = True,
    )

    FILES = parser.add_argument_group(title = "Files IO")
    # input_csv has higher priority than ligand and receptor
    FILES.add_argument(
        '-i',
        '--input_csv',
        type=_path,
        default=None,
        help='The input can be csv file with columns including '
             'protein, protein_name, ligand, ligand_name and complex_name, '
             'and either crystal_ligand or center.',
    )
    FILES.add_argument(
        '-l',
        '--ligand',
        type=_path,
        nargs='*',
        default=[],
        help='The directory to ready-to-dock molecules '
             'or multiple specified molecule files. '
             'Note that current version Only supports SDF file with 3D conformers.'
    )
    FILES.add_argument(
        '-p',
        '--receptor',
        type=_path,
        nargs='*',
        default=[],
        help='The directory to protein receptors '
             'or multiple specified receptor PDB files. '
             'Note that DiffBindFR allows missing pocket side chain atoms rather than backbone atoms. '
             'Users should be careful about missing backbone atoms and fix them before feeding into DiffBindFR.'
    )
    FILES.add_argument(
        '-o',
        '--export_dir',
        type = lambda x: Path(x).absolute(),
        default = '.',
        help = 'The directory where the docked poses will be saved'
    )
    FILES.add_argument(
        '-cfg',
        '--config',
        type = _path,
        default = Path(DiffBindFR.ROOT) / 'configs' / 'diffbindfr_ts.py',
        help = 'The path to global configuration file. Default to ROOT/configs/diffbindfr_ts.py',
    )
    FILES.add_argument(
        '-ckt',
        '--checkpoint',
        type = _path,
        default = Path(DiffBindFR.ROOT) / 'weights' / 'diffbindfr_paper.pth',
        help = "The path to model checkpoint. Default to ROOT/weights/diffbindfr_paper.pth."
    )
    JOBCFG = parser.add_argument_group(title="Job configuration")
    JOBCFG.add_argument(
        '-j',
        '--job',
        type = str,
        choices = ['prep', 'dock'],
        default = 'dock',
        help = 'Running job type. Data preparation and '
               'poses docking using pre-computed data '
               'are included.'
    )
    JOBCFG.add_argument(
        '-np',
        '--num_poses',
        type = int,
        default = 40,
        help = 'The numuber of poses of single protein-molecule pair sampled from model.'
    )
    JOBCFG.add_argument(
        '-dr',
        '--diffbindfr_pocket_radius',
        type=float,
        default=None,
        help='The pocket radius used to modify the default DiffBindFR pocket radius (12A).'
    )
    JOBCFG.add_argument(
        '-mr',
        '--mdn_pocket_radius',
        type=float,
        default=12.0,
        help='The pocket radius used to define the MDN pocket radius (default to 12A).'
    )
    JOBCFG.add_argument(
        '-s',
        '--start',
        type = int,
        default = None,
        help = 'The start of tasks. Note that this follow python slicing rule, starting from 0',
    )
    JOBCFG.add_argument(
        '-e',
        '--end',
        type = int,
        default = None,
        help = 'The end of tasks. Note that this follow python slicing rule.',
    )
    JOBCFG.add_argument(
        '-int',
        '--interval',
        type = int,
        default = None,
        help = 'Tasks start-end interval. '
               'If end is None and interval is not None, apply it.',
    )
    JOBCFG.add_argument(
        '-es',
        '--export_pocket',
        action = 'store_true',
        default = False,
        help = 'Whether export the pockets.',
    )
    JOBCFG.add_argument(
        '-no_ec',
        '--no_error_correction',
        action='store_true',
        default=False,
        help='Whether not to perform error correction resulting from '
             'torsion and rotation entanglement.',
    )
    JOBCFG.add_argument(
        '-no_score',
        '--no_mdn_scoring',
        action='store_true',
        default=False,
        help='Whether not to perform DL based pose scoring using MDN.',
    )
    ct = time.localtime()
    JOBCFG.add_argument(
        '-n',
        '--experiment_name',
        type = str,
        default = f'DiffBindFR_{ct.tm_year}{ct.tm_mon}{ct.tm_mday}{ct.tm_hour}{ct.tm_min}',
        help = 'The name of this flexible docking experiment.'
    )
    JOBCFG.add_argument(
        '-st',
        '--show_traj',
        action='store_true',
        help='Whether show the denoising trajectory.'
    )
    JOBCFG.add_argument(
        '-eval',
        '--evaluation',
        action='store_true',
        default=False,
        help='Evaluate docking performance using ligand columns of job dataframe.'
    )
    JOBCFG.add_argument(
        '-rp',
        '--report_performance',
        action='store_true',
        default=False,
        help='Report the evaluation results about L-RMSD, Centroid, sc-RMSD. '
             'Suppressed when argument "evaluation" is set to False.'
    )
    JOBCFG.add_argument(
        '-cl',
        '--cleanup',
        action='store_true',
        default=False,
        help='Clean up the dataset file (*.lmdb, *.pt) and model_output.pt.'
    )
    RUNCFG = parser.add_argument_group(title="Running configuration")
    RUNCFG.add_argument(
        '-gpu',
        '--gpu_id',
        type = int,
        default = 0,
        help = 'The GPU device ID for single-card evaluation.'
    )
    RUNCFG.add_argument(
        '-cpu',
        '--num_workers',
        type = int,
        default = mp.cpu_count() // 2,
        help = 'The number of workers for multi-processing data preparation or dataloader setting. '
               'Defaults to the half number of available cpus.'
    )
    RUNCFG.add_argument(
        '-bs',
        '--batch_size',
        type = int,
        default = None,
        help='The batch size of dataloader.'
    )
    RUNCFG.add_argument(
        '-v',
        '--verbose',
        action = 'store_true',
        help = 'Whether show the progress bar.'
    )
    RUNCFG.add_argument(
        '-ov',
        '--override',
        action = 'store_true',
        help = 'Whether override the model output.'
    )
    RUNCFG.add_argument(
        '-sd',
        '--seed',
        type=int,
        default=None,
        help='Seed for reproducibility.',
    )
    RUNCFG.add_argument(
        '--debug',
        action='store_true',
        help='Use debug mode to print detailed stdout.'
    )
    return parser

def benchmark_parse_args():
    parser = parse_args()
    BENCH = parser.add_argument_group(title="Benchmark")
    BENCH.add_argument(
        '-d',
        '--data_dir',
        type = _path,
        required = True,
        help = 'The directory to CrossDock-like benchmark dataset. '
               'Every crossdock subset must have the subdirectory about CrossDock ID. '
               'Every subdirectory must have protein.pdb and ligand.sdf (optional for box.csv).'
    )
    BENCH.add_argument(
        '-lb',
        '--lib',
        type = str,
        required = True,
        help = 'The benchmark dataset subset (subdirectory) in the data_dir.'
    )

    return parser

def report_args(args):
    from prettytable import PrettyTable

    ptb = PrettyTable()

    ptb.title = '\033[5;36mCommand Line Parameter\033[0m'
    ptb.field_names = ['\033[31mKeyWord\033[0m', '\033[34mValue\033[0m']
    ptb.align["\033[31mKeyWord\033[0m"] = "l"
    ptb.align["\033[34mValue\033[0m"] = "l"

    for k, v in vars(args).items():
        if isinstance(v, Sequence) and not isinstance(v, str):
            v = ','.join(map(str, v))
        ptb.add_row([f"\033[31m{k}\033[0m", f'\033[34m{v}\033[0m'])

    print(ptb, flush = True)

    return

if __name__ == '__main__':
    parser = benchmark_parse_args()
    args = parser.parse_args()

    report_args(args)

    # print('check get None:', vars(args).get('none', 'None'))
