# Copyright (c) MDLDrugLib. All rights reserved.
import logging

logging.basicConfig(
    level = logging.INFO,
    filename = None,
    datefmt = '%Y/%m/%d %H:%M:%S',
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(
    name = 'TMalign',
)

def save_pdb_without_ter(cmd, filename, selection, *args, **kwargs):
    """
    Save PDB file without TER records. External applications like TMalign and
    DynDom stop reading PDB files at TER records, which might be undesired in
    case of missing loops.
    """
    v = cmd.get_setting_boolean('pdb_use_ter_records')
    if v: cmd.set('pdb_use_ter_records', 0)
    cmd.save(filename, selection, *args, **kwargs)
    if v: cmd.set('pdb_use_ter_records')


def tmalign2(
        cmd,
        mobile,
        target,
        mobile_state=1,
        target_state=1,
        args='',
        exe='TMalign',
        ter=0,
        transform=1,
        object=None,
        quiet=0,
):
    """
    TMalign wrapper. You may also use this as a TMscore or MMalign wrapper
    if you privide the corresponding executable with the "exe" argument.
    Reference: Y. Zhang and J. Skolnick, Nucl. Acids Res. 2005 33, 2302-9
    http://zhanglab.ccmb.med.umich.edu/TM-align/
    Args:
    mobile, target = string: atom selections
    mobile_state, target_state = int: object states {default: 1}
    args = string: Extra arguments like -d0 5 -L 100
    exe = string: Path to TMalign (or TMscore, MMalign) executable
        {default: TMalign}
    ter = 0/1: If ter=0, then ignore chain breaks because TMalign will stop
        at first TER record {default: 0}
    E.g.:
        >>> score = tmalign2(
                cmd,
                'Mobile_RROT',
                'Target_RROT',
                mobile_state = 1,
                target_state = 1,
                args = '',
                exe = 'TMalign',
                ter = 0,
                transform = 1,
                object = 'algnobj',
                quiet = 0,
        )
        score: tmscore1, tmscore2, rmsd
    """
    import subprocess, tempfile, os, re

    ter, quiet = int(ter), int(quiet)

    mobile_filename = tempfile.mktemp('.pdb', 'mobile')
    target_filename = tempfile.mktemp('.pdb', 'target')
    matrix_filename = tempfile.mktemp('.txt', 'matrix')
    mobile_ca_sele = '(%s) and (not hetatm) and name CA and alt +A' % (mobile)
    target_ca_sele = '(%s) and (not hetatm) and name CA and alt +A' % (target)

    if ter:
        save = cmd.save
    else:
        save = save_pdb_without_ter
    save(cmd, mobile_filename, mobile_ca_sele, state=mobile_state)
    save(cmd, target_filename, target_ca_sele, state=target_state)

    exe = cmd.exp_path(exe)
    args = [exe, mobile_filename, target_filename, '-m', matrix_filename] + args.split()

    try:
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE,
            universal_newlines=True)
        lines = process.stdout.readlines()
    except OSError:
        raise Exception('Cannot execute "%s", please provide full path to TMscore or TMalign executable' % (exe))
    finally:
        os.remove(mobile_filename)
        os.remove(target_filename)

    # TMalign >= 2012/04/17
    if os.path.exists(matrix_filename):
        lines += open(matrix_filename).readlines()
        os.remove(matrix_filename)

    r = None
    r_list = list()
    re_score = re.compile(r'TM-score\s*=\s*(\d*\.\d*)')
    rowcount = 0
    matrix = []
    line_it = iter(lines)
    headercheck = False
    alignment = []
    for line in line_it:
        if 4 >= rowcount > 0:
            if rowcount >= 2:
                a = list(map(float, line.split()))
                matrix.extend(a[2:5])
                matrix.append(a[1])
            rowcount += 1
        elif not headercheck and line.startswith(' * '):
            a = line.split(None, 2)
            if len(a) == 3:
                headercheck = a[1]
        elif line.lower().startswith(' -------- rotation matrix'):
            rowcount = 1
        elif line.startswith('(":" denotes'):
            alignment = [next(line_it).rstrip() for _ in range(3)]
        else:
            match = re_score.search(line)
            if match is not None:
                r = float(match.group(1))
                r_list.append(str(r))
        if not quiet:
            logger.info(line.rstrip())

    if not quiet:
        for i in range(0, len(alignment[0]) - 1, 78):
            for line in alignment:
                logger.info(line[i:i + 78])
            logger.info('')

    assert len(matrix) == 3 * 4
    matrix.extend([0, 0, 0, 1])

    if int(transform):
        for model in cmd.get_object_list('(' + mobile + ')'):
            cmd.transform_object(model, matrix, state=0, homogenous=1)

    # alignment object
    if object is not None:
        mobile_idx, target_idx = [], []
        space = {'mobile_idx': mobile_idx, 'target_idx': target_idx}
        cmd.iterate_state(
            mobile_state, mobile_ca_sele,
            'mobile_idx.append("%s`%d" % (model, index))',
            space=space)
        cmd.iterate_state(
            target_state, target_ca_sele,
            'target_idx.append("%s`%d" % (model, index))',
            space=space)
        for i, aa in enumerate(alignment[0]):
            if aa == '-':
                mobile_idx.insert(i, None)
        for i, aa in enumerate(alignment[2]):
            if aa == '-':
                target_idx.insert(i, None)
        if (len(mobile_idx) == len(target_idx) == len(alignment[2])):
            rmsd = cmd.rms_cur(
                ' '.join(idx for (idx, m) in zip(mobile_idx, alignment[1]) if m in ':.'),
                ' '.join(idx for (idx, m) in zip(target_idx, alignment[1]) if m in ':.'),
                cycles=0, matchmaker=4, object=object)
            r_list.append(str(rmsd))
        else:
            logger.info('Could not load alignment object')

    if not quiet:
        if headercheck:
            logger.info('Finished Program:', headercheck)
        if r is not None:
            logger.info('Found in output TM-score = %.4f' % (r))

    return r_list