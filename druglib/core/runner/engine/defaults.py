# Copyright (c) MDLDrugLib. All rights reserved.
import argparse
from druglib import DictAction


def default_argument_parser():
    parser = argparse.ArgumentParser(description="Base Trainer in MDLDrugLib")
    parser.add_argument('config', help = 'Train Config File Path.')
    parser.add_argument('--work-dir', help = 'The Dir to Save logs and model.')
    parser.add_argument('--resume-from', help = 'The Checkpoint File to Resume From.')

    parser.add_argument('--auto-resume', action = 'store_true', help = 'Resume From The Latest Checkpoint Automatically.')
    parser.add_argument('--no-validate', action='store_true', help='Whether or Not to Evaluate The Checkpoint During Training.')
    parser.add_argument('--gpu-id', type = int, default = 0, help = 'ID of GPU to Use (Only Applicable to Non-distributed Training).')
    parser.add_argument('--seed', type = int, default = None, help = 'Random Seed for Experiment Reproduction.')
    parser.add_argument('--diff-seed', action='store_true',
                        help='Whether or Not to Set Different Seeds For Different Ranks.')
    parser.add_argument('--deterministic', action='store_true',
                        help='Whether to Set Deterministic Options For CUDNN Backend.')
    parser.add_argument('--cfg-options', nargs = '+', action = DictAction,
                        help = 'override some settings in the used config, the key-value pair '
                               'in xxx=yyy format will be merged into config file. If the value to '
                               'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                               'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                               'Note that the quotation marks are necessary and that no white space '
                               'is allowed.')
    parser.add_argument(
        '--launcher',
        choices = ['none', 'pytorch', 'slurm', 'mpi'],
        default = 'none',
        help = 'Job Launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    return parser



