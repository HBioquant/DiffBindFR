# Copyright (c) MDLDrugLib. All rights reserved.
import subprocess
import os.path as osp

# User defined Bin path
this_file = osp.abspath(__file__)
this_dir = osp.dirname(this_file)
MSMS_bin = osp.join(this_dir, 'msms')


# automatic detection
if MSMS_bin is None or not osp.exists(MSMS_bin):
    MSMS_bin = 'msms'
    p = subprocess.Popen(
        ["which", MSMS_bin],
        universal_newlines = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE
    )
    MSMS_bin, err = p.communicate()
    MSMS_bin = MSMS_bin.strip()

    if MSMS_bin:
        this_file = osp.abspath(__file__)
        subprocess.check_call(f'sed -i "s/\r$//" {this_file}', shell=True)
        p = subprocess.Popen(
            "awk '/MSMS_bin = /{print NR; exit}' " + this_file,
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
            f'sed -i "{row_number}c MSMS_bin = ' + f"'{MSMS_bin}'" + f'" {this_file}',
            shell = True,
        )
    else:
        MSMS_bin = None

__all__ = ['MSMS_bin']
