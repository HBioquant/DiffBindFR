# Copyright (c) MDLDrugLib. All rights reserved.
import os

def isExecutable(path):
    """Returns true if *path* is an executable."""

    return (isinstance(path, str) and os.path.exists(path) and
        os.access(path, os.X_OK))

def which(program):
    """
    This function is based on the example in:
    http://stackoverflow.com/questions/377017/
    """
    fpath, fname = os.path.split(program)
    fname, fext = os.path.splitext(fname)

    if fpath and isExecutable(program):
        return program
    else:
        if os.name == 'nt' and fext == '':
            program += '.exe'
        for path in os.environ["PATH"].split(os.pathsep):
            path = os.path.join(path, program)
            if isExecutable(path):
                return path
    return None