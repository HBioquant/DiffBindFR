# Copyright (c) MDLDrugLib. All rights reserved.
import os
import subprocess
import warnings
from typing import Tuple, AnyStr, Optional
from packaging.version import parse


def digit_version(
        version_str:str,
        length:int = 4,
) -> Tuple[int]:
    """
    Convert a version string into a tuple of integers
    This method is usually used for comparing two versions.
    For pre-release version: alpha < beta < rc.
    Args:
        version_str:str: The version string.
        length:int: The maximum number of version levels. Defaults to 4,
    Returns:
        Tuple[int]: The version info in digits [integers].
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f"failed to parse version {version}"
    release = list(version.release)#'0.24.4.rc' -> version.release -> (0, 24, 4)
    release = release[:length]
    if len(release) < length:
        release += [0] * (length - len(release))
    if version.is_prerelease:#'0.24.4.rc' -> version.is_prerelease -> True, '0.24.4' -> version.is_prerelease -> False
        mapping = {'a':-3, 'b':-2, 'rc':-1}
        val = -4
        # version.pre can be None
        if version.pre:#'0.24.4.rc' -> version.pre -> ('rc', 0), '0.24.4.rc2' -> version.pre -> ('rc', 2)
            if version.pre[0] not in mapping:
                warnings.warn(f'Unknown prerelease version {version.pre[0]}, '
                              f'version checking may go wrong.')
            else:
                val = mapping[version.pre[0]]
        else:
            release.extend([val, 0])

    elif version.is_postrelease:#'0.24.4-2022' -> version.post -> 2022, '0.24.4.rc2' -> version.post -> None
        release.extend([1, version.post])
    else:
        release.extend([0, 0])

    return tuple(release)


def _minimal_ext_cmd(
        cmd,
) -> AnyStr:
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    env['LANGUAGE'] = 'C'#LANGUAGE is used on win32
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, env = env
    ).communicate()[0]

    return out


def get_git_hash(
        fallback:str = 'unknown',
        digits:Optional[int] = None,
) -> str:
    """
    Get the git hash of the current repo.

    Args:
        fallback:str:: The fallback string when git hash is
            unavailable. Defaults to 'unknown'.
        digits:Optional[int]: Kept digits of the hash. Defaults to None,
            meaning all digits are kept.

    Returns:
        str: Git commit hash.
    """

    if digits is not None and not isinstance(digits, int):
        raise TypeError('digits must be None or an integer')

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
        if digits is not None:
            sha = sha[:digits]
    except OSError:
        sha = fallback

    return sha