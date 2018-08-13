# -*- coding: utf-8 -*-
"""
NiftyNet launch configuration
"""


try:
    import ConfigParser as configparser
except ImportError:
    import configparser


class NiftyNetLaunchConfig(configparser.ConfigParser):
    """Launch configuration settings"""

    pass
