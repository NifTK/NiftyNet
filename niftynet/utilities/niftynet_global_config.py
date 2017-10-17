# -*- coding: utf-8 -*-

from os.path import (expanduser, join, abspath, dirname, isdir, isfile)
from os import makedirs
try:
    from configparser import (ConfigParser, Error)
except ImportError:
    from ConfigParser import (ConfigParser, Error)
from niftynet.utilities.decorators import singleton


@singleton
class NiftyNetGlobalConfig(object):
    """Global configuration settings"""

    def __init__(self):
        self._download_server_url = \
            'https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer'
        self._config_home = join(expanduser('~'), '.niftynet')
        self._config_file = join(self._config_home, 'config.ini')

        config_opts = self.__load_or_create(self._config_file)
        self._niftynet_home = expanduser(config_opts['home'])

    def __load_or_create(self, config_file):
        """Load passed configuration file, if it exists; create a default
        otherwise.

        :param config_file: no sanity checks are performed, as this
        method is for internal use only
        :type config_file: `os.path`
        :returns: a dictionary of parsed configuration options
        :rtype: `dict`
        """
        backup = False
        create_new = False
        global_tag = 'global'
        config_dir = dirname(config_file)
        home_tag = 'home'
        if not isdir(config_dir):
            makedirs(config_dir)
        if isfile(config_file):
            try:
                config = ConfigParser()
                config.read(config_file)
                if global_tag in config:
                    if home_tag in config[global_tag]:
                        return dict(config[global_tag])
                    else:
                        backup = True
                else:
                    backup = True
            except Error:
                backup = True
        else:
            create_new = True

        if backup:
            # TODO
            pass

        if backup or create_new:
            # TODO
            pass

        config = ConfigParser()
        config[global_tag] = {
            'home': '~/niftynet'
        }
        with open(config_file, 'w') as new_config_file:
            config.write(new_config_file)
        return dict(config['global'])

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
