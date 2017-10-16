# -*- coding: utf-8 -*-

from os.path import (expanduser, join, abspath)
from niftynet.utilities.decorators import singleton


@singleton
class NiftyNetGlobalConfig(object):
    """Global configuration settings"""

    def __init__(self):
        self._download_server_url = \
            'https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer'
        self._config_home = join(expanduser('~'), '.niftynet')
        self._config_file = join(self._config_home, 'config.ini')

        self._niftynet_home = abspath(config_opts['home'])

    def __load_or_create(self, config_file):
        """Load passed configuration file, if it exists; create a default
        otherwise.

        :param config_file: no sanity checks are performed, as this
        method is for internal use only
        :type config_file: `os.path`
        :returns: a dictionary of parsed configuration options
        :rtype: `dict`
        """
        # TODO
        pass

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
