# -*- coding: utf-8 -*-
"""
NiftyNet user folder configuration
"""

import sys
from random import choice
from string import ascii_lowercase
from time import strftime

import os
from os.path import expanduser, join, split, isdir, isfile, splitext

# pylint: disable=wrong-import-order
try:
    from configparser import (ConfigParser, Error)
except ImportError:
    from ConfigParser import (ConfigParser, Error)
from niftynet.utilities.decorators import singleton

CONFIG_HOME_VAR = 'niftynet_config_home'
MODEL_ZOO_VAR = 'niftynet_model_zoo'
# SERVER_URL = 'https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer'
SERVER_URL = 'https://github.com/NifTK/NiftyNetModelZoo'


@singleton
class NiftyNetGlobalConfig(object):
    """Global configuration settings"""

    global_section = 'global'
    home_key = 'home'

    niftynet_exts = {'extensions': ['network']}

    def __init__(self):
        self._download_server_url = SERVER_URL
        self._config_home = join(expanduser('~'), '.niftynet')

        try:
            if os.environ[MODEL_ZOO_VAR]:
                self._download_server_url = os.environ[MODEL_ZOO_VAR]
        except KeyError:
            pass

        try:
            if os.environ[CONFIG_HOME_VAR]:
                self._config_home = os.environ[CONFIG_HOME_VAR]
        except KeyError:
            pass

        self._config_file = join(self._config_home, 'config.ini')

        self._niftynet_home = self._config_home

        self.setup()

    def setup(self):
        """
        Read variables from system environment and make directories.

        :return:
        """
        config_opts = NiftyNetGlobalConfig.__load_or_create(self._config_file)
        self._niftynet_home = expanduser(
            config_opts[NiftyNetGlobalConfig.global_section][
                NiftyNetGlobalConfig.home_key])

        if not isdir(self._niftynet_home):
            os.makedirs(self._niftynet_home)

            # create folders for user-defined extensions such as new networks
            for ext in list(NiftyNetGlobalConfig.niftynet_exts):
                extension_subfolder = join(self._niftynet_home, ext)
                NiftyNetGlobalConfig.__create_module(extension_subfolder)
                for mod in NiftyNetGlobalConfig.niftynet_exts[ext]:
                    extension_subsubfolder = join(self._niftynet_home, ext, mod)
                    NiftyNetGlobalConfig.__create_module(extension_subsubfolder)

        for ext in list(NiftyNetGlobalConfig.niftynet_exts):
            extension_subfolder = join(self._niftynet_home, ext)
            sys.path.insert(1, extension_subfolder)
        sys.path.insert(1, self._niftynet_home)
        return self

    @staticmethod
    def __create_module(path):
        """Create the passed path, i.e. folder and place an empty
        ``__init__.py`` file inside.

        :param path: assumed not to exist
        :type path: `os.path`
        """
        os.makedirs(path)
        open(join(path, '__init__.py'), 'a').close()

    @staticmethod
    def __load_or_create(config_file):
        """Load passed configuration file, if it exists; create a default
        otherwise. If this method finds an incorrect config file, it
        backs the file up with a human-readable timestamp suffix and
        creates a default one.

        :param config_file: no sanity checks are performed, as this
            method is for internal use only
        :type config_file: `os.path`
        :returns: a dictionary of parsed configuration options
        :rtype: `dict`
        """
        required_sections = [NiftyNetGlobalConfig.global_section]
        required_keys = {
            required_sections[0]: [NiftyNetGlobalConfig.home_key]
        }
        default_values = {
            required_sections[0]: {
                NiftyNetGlobalConfig.home_key: '~/niftynet'
            }
        }

        backup = False
        if isfile(config_file):
            try:
                config = ConfigParser()
                config.read(config_file)

                # check all required sections and keys present
                for required_section in required_sections:
                    if required_section not in config:
                        backup = True
                        break

                    for required_key in required_keys[required_section]:
                        if required_key not in config[required_section]:
                            backup = True
                            break

                    if backup:
                        break

            except Error:
                backup = True

            if not backup:  # loaded file contains all required
                # config options: so return
                return dict(config)

        config_dir, config_filename = split(config_file)
        if not isdir(config_dir):
            os.makedirs(config_dir)

        if backup:  # config file exists, but does not contain all required
            # config opts: so backup not to override
            timestamp = strftime('%Y-%m-%d-%H-%M-%S')
            random_str = ''.join(choice(ascii_lowercase) for _ in range(3))
            backup_suffix = '-'.join(['backup', timestamp, random_str])

            filename, extension = splitext(config_filename)
            backup_filename = ''.join([filename, '-', backup_suffix, extension])
            backup_file = join(config_dir, backup_filename)
            os.rename(config_file, backup_file)

        # create a new default global config file
        config = ConfigParser(default_values)
        for required_section in required_sections:
            for required_key in required_keys[required_section]:
                config.add_section(required_section)
                config[required_section][required_key] = \
                    default_values[required_section][required_key]
        with open(config_file, 'w') as new_config_file:
            config.write(new_config_file)
        return dict(config)

    def get_niftynet_home_folder(self):
        """Return the folder containing NiftyNet models and data"""
        return self._niftynet_home

    def get_niftynet_config_folder(self):
        """Return the folder containing NiftyNet global configuration"""
        return self._config_home

    def get_default_examples_folder(self):
        """Return the default folder containing NiftyNet examples"""
        return join(self._niftynet_home, 'examples')

    def get_download_server_url(self):
        """Return the URL to the NiftyNet examples server"""
        return self._download_server_url
