# Copyright (c) MDLDrugLib. All rights reserved.
import os, hashlib, gzip, shutil, zipfile, tarfile, contextlib, tempfile
from typing import Optional
from six.moves.urllib.request import urlretrieve
from .logger import get_logger
from .path import mkdir_or_exists

logger = get_logger(
    name=__name__,
)

def compute_md5(
        file_name:str,
        chunk_siz:int=65536
) -> str:
    """
    Compute MD5 of the file.
    Args:
        file_name:str: file name
        chunk_siz:int: chunk size for reading large files
    Returns:
        hash code: str
    """
    md5 = hashlib.md5()
    with open(file_name, 'rb') as f:
        chunk = f.read(chunk_siz)
        while chunk:
            md5.update(chunk)
            chunk = f.read(chunk_siz)
    return md5.hexdigest()

def get_line_count(
        filename:str,
        chunk_size:int=8192*1024
) -> int:
    """
    Get the number of lines in a file.

    Args:
        filename:str: file name.
        chunk_size:int: chunk size for reading large files
    Returns:
        counts:int: lines counts.
    """
    count = 0
    with open(filename, 'rb') as f:
        chunk = f.read(chunk_size)
        while chunk:
            count += chunk.count(b"\n")
            chunk = f.read(chunk_size)
    return count

def download(
        url:str,
        path:str,
        save_file:Optional[str]=None,
        md5:Optional[str]=None
) -> str:
    """
    Download a file from the specified url.
    Skip the downloading step if there exists a file satisfying the given MD5.

    Args:
         url:str: URL to download.
         path:str: Path to save the downloaded file.
         save_file:Optional[str]: Name of save file. if not specified, infer the file from the URL.
         md5:Optional[str]: MD5 of the file.
    Returns:
         save file
    """

    if save_file is None:
        save_file = os.path.basename(url)
        if "?" in save_file:
            save_file = save_file[:save_file.find("?")]
    save_file = os.path.join(path, save_file)
    mkdir_or_exists(save_file)

    if not os.path.exists(save_file) or compute_md5(save_file) != md5:
        logger.info("Downloading %s to %s" % (url, save_file))
        urlretrieve(url, save_file)
    return save_file

def extract(
        zip_file:str,
        member:Optional[str] = None
) -> str:
    """
    Extract file from a zip file. Currently, ``zip``, ``gz``, ``tar.gz``,
    ``tar``, ``tgz`` file are supported.
    Args:
        zip_file:str: zipped file name.
        member:Optional[str]: Extract a specific member from the zip file.
            if not specified, extract all members.
    Returns:

    """
    zip_name, extension = os.path.splitext(zip_file)
    # process ``tar.gz`` file type
    if zip_name.endswith(".tar"):
        extension = ".tar" + extension
        zip_name = zip_name[:-4]

    # extract all members or some member.
    if member is None:
        save_file = zip_name
    else:
        save_file = os.path.join(os.path.dirname(zip_file),
                                 os.path.basename(member))
    if os.path.exists(save_file):
        return save_file

    if member is None:
        logger.info(
            "Extracting %s to %s" % (zip_file, save_file)
        )
    else:
        logger.info(
            "Extracting %s from %s to %s" % format(member, zip_file, save_file)
        )

    if extension == ".gz":
        with gzip.open(zip_file, 'rb') as fin, open(save_file, 'wb') as fout:
            shutil.copyfileobj(fin, fout)
    elif extension == ".zip":
        if member is None:
            with zipfile.ZipFile(zip_file) as fin:
                fin.extractall(save_file)
        else:
            with zipfile.ZipFile(zip_file).open(member, 'r') as fin, open(save_file, 'wb') as fout:
                shutil.copyfileobj(fin, fout)
    elif extension in [".tar.gz", ".tgz", ".tar"]:
        if member is None:
            with tarfile.open(zip_file, "r") as fin:
                fin.extractall(save_file)
        else:
            with tarfile.open(zip_file, 'r').extractfile(member) as fin, open(save_file, "wb") as fout:
                shutil.copyfileobj(fin, fout)
    else:
        raise ValueError(
            f"Unsupported file extension `{extension}`"
        )

    return save_file

@contextlib.contextmanager
def tmpdir_manager(
        base_dir: Optional[str] = None,
):
    """Context manager that deletes a temporary directory on exit."""
    temp = tempfile.mkdtemp(dir = base_dir)
    try:
        yield temp
    finally:
        shutil.rmtree(temp, ignore_errors = True)

