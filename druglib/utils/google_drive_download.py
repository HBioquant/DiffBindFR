# Copyright (c) MDLDrugLib. All rights reserved.
import requests
import zipfile
import warnings
from sys import stdout
import os.path as osp
from .path import mkdir_or_exists
from .logger import print_log


class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.
    """

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = 'https://docs.google.com/uc?export=download'

    @staticmethod
    def download_file_from_google_drive(
            file_id: str,
            dest_path: str,
            overwrite: bool = False,
            unzip: bool = False,
            showsize: bool = False
    ):
        """
        Downloads a shared file from google drive into a given folder.
            Optionally unzips it.
        Reference: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
            https://github.com/ndrplz/google-drive-downloader
        Args:
            file_id: str, the file identifier.
                You can obtain it from the sharable link.
            dest_path: str, the destination where to save the downloaded file.
                Must be a path (for example: './downloaded_file.txt')
            overwrite: bool, optional, if True forces re-download and overwrite.
            unzip: bool, optional, if True unzips a file.
                If the file is not a zip file, ignores it.
            showsize: bool, optional, if True print the current download size.
        E.g.
            GoogleDriveDownloader.download_file_from_google_drive(
            file_id='1iytA1n2z4go3uVCwE__vIKouTKyIDjEq',
            dest_path='./data/google_downloader.zip',
            unzip=True)
        """

        destination_directory = osp.dirname(dest_path)
        mkdir_or_exists(destination_directory)

        if not osp.exists(dest_path) or overwrite:

            session = requests.Session()

            print_log('Downloading {} into {}... '.format(file_id, dest_path))
            stdout.flush()

            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)

            current_download_size = [0]
            GoogleDriveDownloader._save_response_content(response, dest_path, showsize, current_download_size)
            print_log('Done.')

            if unzip:
                try:
                    print_log('Unzipping...')
                    stdout.flush()
                    with zipfile.ZipFile(dest_path, 'r') as z:
                        z.extractall(destination_directory)
                    print_log('Done.')
                except zipfile.BadZipfile:
                    warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_id))

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination, showsize, current_size):
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    if showsize:
                        print_log('\r' + GoogleDriveDownloader.sizeof_fmt(current_size[0]))
                        stdout.flush()
                        current_size[0] += GoogleDriveDownloader.CHUNK_SIZE

    @staticmethod
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return '{:.1f} {}{}'.format(num, unit, suffix)
            num /= 1024.0
        return '{:.1f} {}{}'.format(num, 'Yi', suffix)