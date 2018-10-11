#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Downloading model zoo specifications and model zoo contents.
"""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import math
import os
import re
import shutil
import tarfile
import tempfile
from distutils.version import LooseVersion

# pylint: disable=wrong-import-order
from six.moves.urllib.parse import urlparse
from six.moves.urllib.request import urlopen
from six.moves.urllib.request import urlretrieve

# Used with the min_download_api settings option to determine
# if the downloaded configuration file is compatible with
# this version of NiftyNet downloader code
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig
from niftynet.utilities import NiftyNetLaunchConfig
from niftynet.utilities.util_common import print_progress_bar
from niftynet.utilities.versioning import get_niftynet_version, \
    get_niftynet_version_string

DOWNLOAD_API_VERSION = "1.0"
CONFIG_FILE_EXT = ".ini"


def download(example_ids,
             download_if_already_existing=False,
             verbose=True):
    """
    Downloads standard NiftyNet examples such as data, samples

    :param example_ids:
        A list of identifiers for the samples to download
    :param download_if_already_existing:
        If true, data will always be downloaded
    :param verbose:
        If true, download info will be printed
    """

    global_config = NiftyNetGlobalConfig()

    config_store = ConfigStore(global_config, verbose=verbose)

    # If a single id is specified, convert to a list
    example_ids = [example_ids] \
        if not isinstance(example_ids, (tuple, list)) else example_ids

    if not example_ids:
        return False

    # Check if the server is running by looking for a known file
    remote_base_url_test = raw_file_url(
        global_config.get_download_server_url())
    server_ok = url_exists(remote_base_url_test)
    if verbose:
        print("Accessing: {}".format(global_config.get_download_server_url()))

    any_error = False
    for example_id in example_ids:
        if not example_id:
            any_error = True
            continue
        if config_store.exists(example_id):
            update_ok = config_store.update_if_required(
                example_id,
                download_if_already_existing)
            any_error = (not update_ok) or any_error
        else:
            any_error = True
            if server_ok:
                print(example_id + ': FAIL. ')
                print('No NiftyNet example was found for ' + example_id + ".")

    # If errors occurred and the server is down report a message
    if any_error and not server_ok:
        print("The NiftyNetExamples server is not running")

    return not any_error


def download_file(url, download_path):
    """
    Download a file from a resource URL to the given location

    :param url: URL of the file to download
    :param download_path: location where the file should be saved
    """
    # Extract the filename from the URL
    filename = os.path.basename(download_path)

    # Ensure the output directory exists
    output_directory = os.path.dirname(download_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get a temporary file path for the compressed file download
    temp_folder = tempfile.mkdtemp()
    downloaded_file = os.path.join(temp_folder, filename)

    # Download the file
    urlretrieve(url, downloaded_file, reporthook=progress_bar_wrapper)

    # Move the file to the destination folder
    shutil.move(downloaded_file, download_path)
    shutil.rmtree(temp_folder, ignore_errors=True)


def download_and_decompress(url, download_path, verbose=True):
    """
    Download an archive from a resource URL and
    decompresses/unarchives to the given location

    :param url: URL of the compressed file to download
    :param download_path: location where the file should be extracted
    """

    # Extract the filename from the URL
    print('Downloading from {}'.format(url))
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)

    # Ensure the output directory exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Get a temporary file path for the compressed file download
    temp_folder = tempfile.mkdtemp()
    downloaded_file = os.path.join(temp_folder, filename)

    # Download the file
    if verbose:
        urlretrieve(url, downloaded_file, reporthook=progress_bar_wrapper)
    else:
        urlretrieve(url, downloaded_file)

    # Decompress and extract all files to the specified local path
    tar = tarfile.open(downloaded_file, "r")
    tar.extractall(download_path)
    tar.close()

    # Remove the downloaded file
    shutil.rmtree(temp_folder, ignore_errors=True)


class ConfigStore(object):
    """
    Manages a configuration file store based on a
    remote repository with local caching
    """

    def __init__(self, global_config, verbose=True):
        self._download_folder = global_config.get_niftynet_home_folder()
        self._config_folder = global_config.get_niftynet_config_folder()
        self._local = ConfigStoreCache(
            os.path.join(self._config_folder, '.downloads_local_config_cache'))
        self._remote = RemoteProxy(self._config_folder,
                                   global_config.get_download_server_url())
        self._verbose = verbose

    def exists(self, example_id):
        """
        Returns True if a record exists for this example_id,
        either locally or remotely
        """

        return self._local.exists(example_id) or self._remote.exists(example_id)

    # pylint: disable=broad-except
    def update_if_required(self,
                           example_id,
                           download_if_already_existing=False):
        """
        Downloads data using the configuration file
        if it is not already up to date.
        Returns True if no update was required and no errors occurred
        """

        try:
            self._remote.update(example_id)
            remote_update_failed = False
        except Exception as fail_msg:
            print("Warning: updating the examples file "
                  "from the server caused an error: {}".format(fail_msg))
            remote_update_failed = True

        current_config, current_entries = \
            self._local.get_download_params(example_id)
        remote_config, remote_entries = \
            self._remote.get_download_params(example_id)

        if not remote_entries:
            if remote_update_failed:
                print(example_id + ": FAIL.")
                print("Cannot download the examples configuration file. "
                      "Is the server down?")
            else:
                print(example_id + ": FAIL. Nothing to download")
            return False

        else:
            # Always download if the local file is empty, or force by arguments
            force_download = download_if_already_existing or \
                             (not current_config and not current_entries)
            data_missing = self._are_data_missing(remote_entries, example_id)
            if force_download or data_missing or self._is_update_required(
                    current_config, remote_config):
                self._check_minimum_versions(remote_config)
                self._download(remote_entries, example_id)
                self._replace_local_with_remote(example_id)
            else:
                print(example_id + ": OK. ")
                print("Already downloaded. "
                      "Use the -r option to download again.")

        return True

    @staticmethod
    def _check_minimum_versions(remote_config):
        # Checks whether a minimum download API is specified
        if 'min_download_api' in remote_config:
            try:
                min_download_api = LooseVersion(
                    remote_config['min_download_api'])
            except AttributeError:
                min_download_api = LooseVersion(DOWNLOAD_API_VERSION)
            current_download_api_version = LooseVersion(DOWNLOAD_API_VERSION)
            if min_download_api > current_download_api_version:
                raise ValueError(
                    "This example requires a newer version of NiftyNet.")

        # Checks whether a minimum NiftyNet version is specified
        if 'min_niftynet' in remote_config:
            try:
                min_niftynet = LooseVersion(remote_config['min_niftynet'])
                current_version = LooseVersion(get_niftynet_version())
            except AttributeError:
                min_niftynet = LooseVersion('0')
                current_version = LooseVersion('0')
            if min_niftynet > current_version:
                raise ValueError("This example requires NiftyNet "
                                 "version %s or later.", min_niftynet)

    @staticmethod
    def _is_update_required(current_config, remote_config):
        """
        If no version information locally,
        then update only if version information is specified remotely
        We are assuming that this is overridden
        by the case of no local information at all
        """

        if 'version' not in current_config:
            return 'version' in remote_config

        return LooseVersion(current_config['version']) < \
               LooseVersion(remote_config['version'])

    def _download(self, remote_config_sections, example_id):
        for section_name, config_params in remote_config_sections.items():
            if 'action' in config_params:
                action = config_params.get('action').lower()
                if action == 'expand':
                    if 'url' not in config_params:
                        raise ValueError('No URL was found in the download '
                                         'configuration file')
                    local_download_path = self._get_local_download_path(
                        config_params, example_id)
                    download_and_decompress(
                        url=config_params['url'],
                        download_path=local_download_path,
                        verbose=self._verbose)
                    print('{} -- {}: OK.'.format(example_id, section_name))
                    print("Downloaded data to " + local_download_path)
                else:
                    print(example_id + ": FAIL.")
                    print("I do not know the action " + action +
                          ". Perhaps you need to update NiftyNet?")

    def _get_local_download_path(self, remote_config, example_id):
        destination = remote_config.get('destination', 'examples')
        local_id = remote_config.get('local_id', example_id)
        return os.path.join(self._download_folder, destination, local_id)

    def _replace_local_with_remote(self, example_id):
        local_filename = self._local.get_local_path(example_id)
        remote_filename = self._remote.get_local_path(example_id)
        shutil.copyfile(remote_filename, local_filename)

    def _are_data_missing(self, remote_config_sections, example_id):
        for _, config_params in remote_config_sections.items():
            if 'action' in config_params:
                action = config_params.get('action').lower()
                if action == 'expand':
                    local_download_path = self._get_local_download_path(
                        config_params, example_id)
                    if not os.path.isdir(local_download_path):
                        return True

                    non_system_files = [
                        f for f in os.listdir(local_download_path)
                        if not f.startswith('.')]
                    if not non_system_files:
                        return True
        return False


class ConfigStoreCache(object):
    """
    A local cache for configuration files
    """

    def __init__(self, cache_folder):
        self._cache_folder = cache_folder
        if not os.path.exists(self._cache_folder):
            os.makedirs(self._cache_folder)

    def exists(self, example_id):
        """
        Returns True if a record exists for this example_id,
        either locally or remotely
        """

        return os.path.isfile(self.get_local_path(example_id))

    def get_local_path(self, example_id):
        """
        Returns the full path to the locally cached configuration file
        """

        return os.path.join(self._cache_folder,
                            example_id + '_main'+ CONFIG_FILE_EXT)
        # return os.path.join(self._cache_folder, example_id + CONFIG_FILE_EXT)

    def get_local_cache_folder(self):
        """
        Returns the folder in which the cached files are stored
        """

        return self._cache_folder

    def get_download_params(self, example_id):
        """
        Returns the local configuration file for this example_id
        """
        config_filename = self.get_local_path(example_id)

        parser = NiftyNetLaunchConfig()
        parser.read(config_filename)
        if parser.has_section('config'):
            config_section = dict(parser.items('config'))
        else:
            config_section = {}

        other_sections = {}
        for section in parser.sections():
            if section != 'config' and section != 'DEFAULT':
                other_sections[section] = dict(parser.items(section))
        return config_section, other_sections


class RemoteProxy(object):
    """
    A remote configuration file store with a local cache
    """

    def __init__(self, parent_store_folder, base_url):
        self._cache = ConfigStoreCache(
            os.path.join(parent_store_folder, '.downloads_remote_config_cache'))
        self._remote = RemoteConfigStore(base_url)

    def exists(self, example_id):
        """
        Returns True if a record exists locally or remotely
        """

        return self._cache.exists(example_id) or self._remote.exists(example_id)

    def update(self, example_id):
        """
        Retrieves the latest record from the remote store and
        puts locally into the remote cache
        """

        download_file(self._remote.get_url(example_id),
                      self._cache.get_local_path(example_id))

    def get_download_params(self, example_id):
        """
        Returns the local configuration file for this example_id
        """

        return self._cache.get_download_params(example_id)

    def get_local_path(self, example_id):
        """
        Returns the full path to the locally cached configuration file
        """

        return self._cache.get_local_path(example_id)


class RemoteConfigStore(object):
    """
    A remote configuration file store
    """

    def __init__(self, base_url):
        self._base_url = base_url

    def exists(self, example_id):
        """
        Returns true if the record exists on the remote server
        """

        return url_exists(self.get_url(example_id))

    def get_url(self, example_id):
        """
        Gets the URL for the record for this example_id
        """
        return raw_file_url(self._base_url, example_id)


def raw_file_url(base_url, example_id=None):
    """
    Returns the url for the raw file on a GitLab server
    """
    _branch_name = '5-reorganising-with-lfs'

    if not example_id:
        return '{}/raw/{}/README.md'.format(base_url, _branch_name)
    example_id = re.sub('_model_zoo', '', example_id, 1)
    return '{}/raw/{}/{}/main{}'.format(
        base_url, _branch_name, example_id, CONFIG_FILE_EXT)
    # return base_url + '/raw/new_dataset_api/' + file_name
    # return base_url + '/raw/master/' + file_name
    # return base_url + '/raw/revising-config/' + file_name


# pylint: disable=broad-except
def url_exists(url):
    """
    Returns true if the specified url exists, without any redirects
    """

    try:
        connection = urlopen(url)
        return connection.getcode() < 400
    except Exception:
        return False


def progress_bar_wrapper(count, block_size, total_size):
    """
    Uses the common progress bar in the urlretrieve hook format
    """
    if block_size * 5 >= total_size:
        # no progress bar for tiny files
        return
    print_progress_bar(
        iteration=count,
        total=math.ceil(float(total_size) / float(block_size)),
        prefix="Downloading (total: %.2f M): " % (total_size * 1.0 / 1e6))


def main():
    """
    Launch download process with user-specified options.

    :return:
    """
    arg_parser = argparse.ArgumentParser(
        description="Download NiftyNet sample data")
    arg_parser.add_argument(
        "-r", "--retry",
        help="Force data to be downloaded again",
        required=False,
        action='store_true')
    arg_parser.add_argument(
        'sample_id',
        nargs='+',
        help="Identifier string(s) for the example(s) to download")
    version_string = get_niftynet_version_string()
    arg_parser.add_argument(
        "-v", "--version",
        action='version',
        version=version_string)
    args = arg_parser.parse_args()

    if not download(args.sample_id, args.retry):
        return -1

    return 0


if __name__ == "__main__":
    main()
