# -*- coding: utf-8 -*-
"""
NiftyNet launch configuration
"""

import os
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
from tensorflow import logging as tf_logging


class NiftyNetLaunchConfig(configparser.ConfigParser):
    """
    Launch configuration settings.

    This class provides the same interface and functionality
    as the built-in `ConfigParser` class, except for the
    constructor, which takes no arguments. Beyond that, it
    can also parse the new YAML configuration format of
    NiftyNet. The former is for backwards compatibility only,
    while the latter is the purpose of this class' life.
    """

    def __init__(self):
        """
        Initialise empty configuration, ready to read in a
        configuration file.
        """

        configparser.ConfigParser.__init__(self)

    def read(self, filenames, encoding=None):

        if not isinstance(filenames, (list, )):
            _filenames = [filenames]
        else:
            _filenames = filenames

        warn_deprecation = False
        for filename in _filenames:
            if os.path.splitext(filename)[1].lower() == '.ini':
                warn_deprecation = True
                break

        if warn_deprecation:
            tf_logging.warn(
                'INI configuration files are deprecated in favor of'
                ' YAML configuration files. Support for INI configuration'
                ' files will be dropped in a future release.'
            )

        return configparser.ConfigParser.read(self, filenames, encoding)

    def sections(self):
        return configparser.ConfigParser.sections(self)

    def items(self, section=None, raw=False, vars=None):
        kwargs = {'vars': vars, 'raw': raw}
        if section is not None:
            kwargs['section'] = section
        return configparser.ConfigParser.items(self, **kwargs)

    def add_section(self, section):
        configparser.ConfigParser.add_section(self, section)

    def set(self, section, option, value=None):
        configparser.ConfigParser.set(self, section, option, value)

    def remove_section(self, section):
        configparser.ConfigParser.remove_section(self, section)

    def has_section(self, section):
        return configparser.ConfigParser.has_section(self, section)
