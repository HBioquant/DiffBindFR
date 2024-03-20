# Copyright (c) MDLDrugLib. All rights reserved.
import subprocess, re
import os.path as osp

# User defined Bin path
this_file = osp.abspath(__file__)
this_dir = osp.dirname(this_file)
DSSP_bin = osp.join(this_dir, 'mkdssp')

# automatic detection
if DSSP_bin is None or not osp.exists(DSSP_bin):
    DSSP_bin = 'dssp'
    if DSSP_bin == "dssp":
        DSSP_bin = "mkdssp"
    elif DSSP_bin == "mkdssp":
        DSSP_bin = "dssp"
    else:
        raise NotImplementedError(DSSP_bin)
    p = subprocess.Popen(
        ["which", DSSP_bin],
        universal_newlines = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    DSSP_bin, err = p.communicate()
    DSSP_bin = DSSP_bin.strip()

    if DSSP_bin:
        subprocess.run(f'sed -i "s/\r$//" {this_file}', shell=True)
        p = subprocess.Popen(
            "awk '/DSSP_bin = /{print NR; exit}' " + this_file,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            output, error = p.communicate()
            row_number = int(output)
        except:
            row_number = 8  # current scripts number

        subprocess.run(
            f'sed -i "{row_number}c DSSP_bin = ' + f"'{DSSP_bin}'" + f'" {this_file}',
            shell=True
        )

try:
    if not DSSP_bin:
        raise ValueError('No DSSP detected.')

    version_string = subprocess.check_output(
        [DSSP_bin, "--version"], universal_newlines = True
    )
    dssp_version = re.search(r"\s*([\d.]+)", version_string).group(1)
except:
    # probably invalid DSSP executable file
    DSSP_bin = None
    dssp_version = ''

__all__ = ['dssp_version', 'DSSP_bin']
