# -*- coding: utf-8 -*-

import os
from os.path import expanduser


class NiftyNetGlobalConfig(object):
    """Global configuration settings"""

    def __init__(self):
        self._download_server_url = \
            'https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer'
        self._config_home = os.path.join(expanduser("~"), '.niftynet')

        # ToDo: fetch NiftyNet home folder from a global configuration file
        self._niftynet_home = os.path.join(expanduser("~"), 'niftynet')

    def get_niftynet_home_folder(self):
        """Return the folder containing NiftyNet models and data"""
        return self._niftynet_home

    def get_niftynet_config_folder(self):
        """Return the folder containing NiftyNet global configuration"""
        return self._config_home

    def get_default_examples_folder(self):
        """Return the default folder containing NiftyNet examples"""
        return os.path.join(self._niftynet_home, 'examples')

    def get_download_server_url(self):
        """Return the URL to the NiftyNet examples server"""
        return self._download_server_url
