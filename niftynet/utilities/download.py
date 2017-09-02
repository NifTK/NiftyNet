#!/usr/bin/python
#  -*- coding: utf-8 -*-

import tarfile
import tempfile
from shutil import copyfile

import six
from six.moves.urllib.request import urlretrieve
from six.moves.urllib.parse import urlparse
from six.moves.urllib.request import urlopen

from os.path import basename

import argparse
import os

from six.moves.configparser import SafeConfigParser

from os.path import expanduser
from distutils.version import LooseVersion

from niftynet.utilities.versioning import get_niftynet_version, get_niftynet_version_string

# Used with the min_download_api settings option to determine if the downloaded configuration file is compatible with
# this version of NiftyNet downloader code
DOWNLOAD_API_VERSION = "1.0"


def download(example_ids, niftynet_base_folder=None, download_if_already_existing=False):
    """
    Downloads standard NiftyNet examples such as data, samples

    :param example_ids: A list of identifiers for the samples to download
    :param niftynet_base_folder: The base folder where downloads are stored
    :param download_if_already_existing: If true, data will always be downloaded
    """

    if not niftynet_base_folder:
        niftynet_base_folder = os.path.join(expanduser("~"), 'niftynet')

    remote_base_url = 'https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer/raw/master/'

    config_store = ConfigStore(niftynet_base_folder, remote_base_url)

    # If a single id is specified, convert to a list
    example_ids = [example_ids] if not isinstance(example_ids, (tuple, list)) else example_ids

    # Check if the server is running by looking for a known file
    remote_base_url_test = remote_base_url + "README.md"
    server_ok = url_exists(remote_base_url_test)

    any_error = False
    for example_id in example_ids:
        if config_store.exists(example_id):
            update_ok = config_store.update_if_required(example_id, download_if_already_existing)
            any_error = (not update_ok) or any_error
        else:
            any_error = True
            if server_ok:
                print(example_id + ': FAIL. No NiftyNet example was found for ' + example_id + ".")

    # If errors occurred and the server is down report a message
    if any_error and not server_ok:
        print("The NiftyNetExamples server is not running")

    return any_error


def download_file(url, download_path):
    """
    Download a file from a resource URL to the given location

    :param url: URL of the file to download
    :param download_path: location where the file should be saved
    """

    # Extract the filename from the URL
    parsed = urlparse(url)
    filename = basename(parsed.path)

    # Ensure the output directory exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Get a temporary file path for the compressed file download
    downloaded_file = os.path.join(tempfile.gettempdir(), filename)

    # Download the file
    urlretrieve(url, downloaded_file)

    # Move the file to the destination folder
    destination_path = os.path.join(download_path, filename)
    os.rename(downloaded_file, destination_path)


def download_and_decompress(url, download_path):
    """
    Download an archive from a resource URL and decompresses/unarchives to the given location

    :param url: URL of the compressed file to download
    :param download_path: location where the file should be extracted
    """

    # Extract the filename from the URL
    parsed = urlparse(url)
    filename = basename(parsed.path)

    # Ensure the output directory exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Get a temporary file path for the compressed file download
    downloaded_file = os.path.join(tempfile.gettempdir(), filename)

    # Download the file
    urlretrieve(url, downloaded_file)

    # Decompress and extract all files to the specified local path
    tar = tarfile.open(downloaded_file, "r")
    tar.extractall(download_path)
    tar.close()

    # Remove the downloaded file
    os.remove(downloaded_file)


class ConfigStore:
    """Manages a configuration file store based on a remote repository with local caching"""

    def __init__(self, parent_store_folder, remote_base_url):
        self._parent_store_folder = parent_store_folder
        self._local = ConfigStoreCache(os.path.join(parent_store_folder, '.downloads_local_config_cache'))
        self._remote = RemoteProxy(parent_store_folder, remote_base_url)

    def exists(self, example_id):
        """Returns True if a record exists for this example_id, either locally or remotely"""

        return self._local.exists(example_id) or self._remote.exists(example_id)

    def update_if_required(self, example_id, download_if_already_existing=False):
        """
        Downloads data using the configuration file if it is not already up to date
        Returns True if no update was required and no errors occurred
        """

        try:
            self._remote.update(example_id)
            remote_update_failed = False
        except Exception as e:
            print("Warning: updating the examples file from the server caused an error: " + str(e))
            remote_update_failed = True

        current_config, current_entries = self._local.get_download_params(example_id)
        remote_config, remote_entries = self._remote.get_download_params(example_id)

        if not remote_entries:
            if remote_update_failed:
                print(example_id + ": FAIL. Cannot download the examples configuration file. Is the server down?")
            else:
                print(example_id + ": FAIL. Nothing to download")
            return False

        else:
            # Always download if the local file is empty, or force by arguments
            force_download = download_if_already_existing or (not current_config and not current_entries)
            data_missing = self._are_data_missing(remote_entries, example_id)
            if force_download or data_missing or self._is_update_required(current_config, remote_config):
                self._check_minimum_versions(remote_config)
                self._download(remote_entries, example_id)
                self._replace_local_with_remote_config(example_id)
            else:
                print(example_id + ": OK. Already downloaded. Use the -r option to download again.")

        return True

    @staticmethod
    def _check_minimum_versions(remote_config):
        # Checks whether a minimum download API is specified
        if 'min_download_api' in remote_config:
            min_download_api = remote_config['min_download_api']
            current_download_api_version = DOWNLOAD_API_VERSION
            if LooseVersion(min_download_api) > LooseVersion(current_download_api_version):
                raise ValueError("This example requires a newer version of NiftyNet.")

        # Checks whether a minimum NiftyNet version is specified
        if 'min_niftynet' in remote_config:
            min_niftynet = remote_config['min_niftynet']
            current_version = get_niftynet_version()
            if LooseVersion(min_niftynet) > LooseVersion(current_version):
                raise ValueError("This example requires NiftyNet version " + min_niftynet + " or later.")

    @staticmethod
    def _is_update_required(current_config, remote_config):

        # If no version information locally, then update only if version information is specified remotely
        # We are assuming that this is overridden by the case of no local information at all
        if 'version' not in current_config:
            return 'version' in remote_config

        else:
            return LooseVersion(current_config['version']) < LooseVersion(remote_config['version'])

    def _download(self, remote_config_sections, example_id):
        for section_name, config_params in remote_config_sections.items():
            if 'action' in config_params:
                action = config_params.get('action').lower()
                if action == 'expand':
                    if 'url' not in config_params:
                        raise ValueError('No URL was found in the download configuration file')
                    local_download_path = self._get_local_download_path(config_params, example_id)
                    download_and_decompress(url=config_params['url'], download_path=local_download_path)
                    print(example_id + ": OK. Downloaded data to " + local_download_path)
                else:
                    print(example_id + ": FAIL. I do not know the action " + action +
                          ". Perhaps you need to update NiftyNet?")

    def _get_local_download_path(self, remote_config, example_id):
        destination = remote_config.get('destination', 'examples')
        local_id = remote_config.get('local_id', example_id)
        return os.path.join(self._parent_store_folder, destination, local_id)

    def _replace_local_with_remote_config(self, example_id):
        local_filename = self._local.get_local_path(example_id)
        remote_filename = self._remote.get_local_path(example_id)
        copyfile(remote_filename, local_filename)

    def _are_data_missing(self, remote_config_sections, example_id):
        for section_name, config_params in remote_config_sections.items():
            if 'action' in config_params:
                action = config_params.get('action').lower()
                if action == 'expand':
                    local_download_path = self._get_local_download_path(config_params, example_id)
                    if not os.path.isdir(local_download_path):
                        return True

                    non_system_files = [f for f in os.listdir(local_download_path) if not f.startswith('.')]
                    if not non_system_files:
                        return True
        return False


class ConfigStoreCache:
    """A local cache for configuration files"""

    def __init__(self, cache_folder):
        self._cache_folder = cache_folder
        if not os.path.exists(self._cache_folder):
            os.makedirs(self._cache_folder)

    def exists(self, example_id):
        """Returns True if a record exists for this example_id, either locally or remotely"""

        return os.path.isfile(self.get_local_path(example_id))

    def get_local_path(self, example_id):
        """Returns the full path to the locally cached configuration file"""

        return os.path.join(self._cache_folder, example_id + '.ini')

    def get_local_cache_folder(self):
        """Returns the folder in which the cached files are stored"""

        return self._cache_folder

    def get_download_params(self, example_id):
        """Returns the local configuration file for this example_id"""

        config_filename = self.get_local_path(example_id)

        parser = SafeConfigParser()
        parser.read(config_filename)
        config_section = dict(parser.items('config')) if 'config' in parser else {}
        other_sections = {key: value for key, value in parser.items() if key != 'config' and key != 'DEFAULT'}
        return config_section, other_sections


class RemoteProxy:
    """A remote configuration file store with a local cache"""

    def __init__(self, parent_store_folder, base_url):
        self._cache = ConfigStoreCache(os.path.join(parent_store_folder, '.downloads_remote_config_cache'))
        self._remote = RemoteConfigStore(base_url)

    def exists(self, example_id):
        """Returns True if a record exists locally or remotely"""

        return self._cache.exists(example_id) or self._remote.exists(example_id)

    def update(self, example_id):
        """Retrieves the latest record from the remote store and puts locally into the remote cache"""

        download_file(self._remote.get_url(example_id), self._cache.get_local_cache_folder())

    def get_download_params(self, example_id):
        """Returns the local configuration file for this example_id"""

        return self._cache.get_download_params(example_id)

    def get_local_path(self, example_id):
        """Returns the full path to the locally cached configuration file"""

        return self._cache.get_local_path(example_id)


class RemoteConfigStore:
    """A remote configuration file store"""

    def __init__(self, base_url):
        self._base_url = base_url

    def exists(self, example_id):
        """Returns true if the record exists on the remote server"""

        return url_exists(self.get_url(example_id))

    def get_url(self, example_id):
        """Gets the URL for the record for this example_id"""

        return six.moves.urllib.parse.urljoin(self._base_url, example_id + '.ini')


def url_exists(url):
    """Returns true if the specified url exists, without any redirects"""

    try:
        connection = urlopen(url)
        return connection.getcode() < 400
    except Exception as e:
        return False


def main():
    arg_parser = argparse.ArgumentParser(description="Download NiftyNet sample data")
    arg_parser.add_argument("-r", "--retry", help="Force data to be downloaded again", required=False,
                            action='store_true')
    arg_parser.add_argument("-d", "--data_folder", help="Change data download location", required=False,
                            default=None)
    arg_parser.add_argument('sample_id', nargs='*', help="Identifier string for the example to download")
    version_string = get_niftynet_version_string()
    arg_parser.add_argument("-v", "--version", action='version', version=version_string)
    args = arg_parser.parse_args()

    if not download(args.sample_id, args.data_folder, args.retry):
        return -1

    return 0

if __name__ == "__main__":
    main()
