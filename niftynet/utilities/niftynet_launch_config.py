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
import yaml


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

        # used both as actual configuration dict, and as flag
        # indicating what file has been used
        self._yaml_dict = dict()

    def read(self, filename, encoding=None):
        """
        Read in given file and store configuration.

        Any newly read-in configuration will have precedence
        over existing configuration.

        :param filename: contrary to `ConfigParser`, this
        method supports reading of only a single file

        :raises ValueError: if newly passed configuration file
        is incompatible with existing configuration, i.e.
        YAML configuration has been read previously, or vice
        versa.
        """

        yaml_ext, ini_ext = '.yml', '.ini'
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext == ini_ext:
            tf_logging.warn(
                'INI configuration files are deprecated in favor of'
                ' YAML configuration files. Support for INI configuration'
                ' files will be dropped in a future release.'
            )
            if self._yaml_dict:
                raise ValueError(
                    'YAML configuration read previously. Refusing to'
                    ' read in INI on top of it.'
                )

            return configparser.ConfigParser.read(self, filename, encoding)

        elif file_ext == yaml_ext:
            if configparser.ConfigParser.sections(self):
                raise ValueError(
                    'INI configuration read previously. Refusing to'
                    ' read in YAML on top of it.'
                )

            with open(filename) as yaml_file:
                self._yaml_dict = yaml.load(yaml_file.read())

            return self._yaml_dict

        else:
            raise ValueError('Configuration file format not recognised.')

    def sections(self):
        if self._yaml_dict:
            return self._yaml_dict.keys()
        else:
            return configparser.ConfigParser.sections(self)

    def items(self, section=None, raw=False, vars=None):
        if self._yaml_dict:
            if section is not None:
                return self._yaml_dict[section]
            else:
                return self._yaml_dict
        else:
            kwargs = {'vars': vars, 'raw': raw}
            if section is not None:
                kwargs['section'] = section
            return configparser.ConfigParser.items(self, **kwargs)

    def add_section(self, section):
        # sanity check
        assert section
        if self._yaml_dict:
            if section not in self._yaml_dict:
                self._yaml_dict[section] = dict()
        else:
            configparser.ConfigParser.add_section(self, section)

    def set(self, section, option, value=None):
        configparser.ConfigParser.set(self, section, option, value)

    def remove_section(self, section):
        configparser.ConfigParser.remove_section(self, section)

    def has_section(self, section):
        return configparser.ConfigParser.has_section(self, section)
