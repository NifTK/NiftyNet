# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re

TRUE_VALUE = {'yes', 'true', 't', 'y', '1'}
FALSE_VALUE = {'no', 'false', 'f', 'n', '0'}
ARRAY_TYPES = {"(": ")", "[": "]"}


def str2boolean(string_input):
    """
    convert user input config string to boolean
    :param string_input: any string in TRUE_VALUE or FALSE_VALUE
    :return: True or False
    """
    if string_input.lower() in TRUE_VALUE:
        return True
    elif string_input.lower() in FALSE_VALUE:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# TODO: passing arrays to application
def numarray(string_input):
    """
    convert user input config string to an array
    The array should be in one of the following form:
    [1, 2, 3, ...]
    (1, 2, 3, ...)
    1, 2, 3, ...
    :param string_input: input string representation of an array
    :return: the array or single value when output length is 1
    """
    if string_input is None or string_input == '':
        raise argparse.ArgumentTypeError('parameter not specified.')
    if string_input[0] in ARRAY_TYPES:
        expected_right_most = ARRAY_TYPES[string_input[0]]
        if not string_input[-1] == expected_right_most:
            raise argparse.ArgumentTypeError(
                'incorrect array format {}'.format(string_input))
        else:
            string_input = string_input[1:-1]
    try:
        array = map(int, string_input.split(','))
    except ValueError:
        try:
            array = map(float, string_input.split(','))
        except ValueError:
            raise argparse.ArgumentTypeError(
                'array expected, unknown array input {}'.format(string_input))
    if len(array) < 1:
        raise argparse.ArgumentTypeError(
            'array expected, unknown array input {}'.format(string_input))
    if len(array) == 1:
        return array[0]
    return tuple(array)


def stringarray(string_input):
    values = string_input.split(',')
    list_of_values = [standardise_string(component.strip())
                      for component in values]
    return tuple(list_of_values)


def standardise_section_name(configparser, old_name):
    """
    rename configparser section
    This helper is useful when user specifies complex section names
    """
    new_name = standardise_string(old_name)
    if old_name == new_name:
        return old_name
    items = configparser.items(old_name)
    configparser.add_section(new_name)
    for (name, value) in items:
        configparser.set(new_name, name, value)
    configparser.remove_section(old_name)
    return new_name


def standardise_string(input_string):
    """
    to make the user's input consistent
    replace any characters not in set [0-9a-zA-Z] with underscrore _

    :param input_string: to be standardised
    :return: capitalised string
    """
    new_name = re.sub('[^0-9a-zA-Z]+', '_', input_string.strip())
    return new_name.upper()


def check_required_sections(config, app_type):
    import niftynet.utilities.user_parameters_parser as param_parser
    required_custom_section = standardise_string(
        param_parser.CUSTOM_SECTIONS.get(app_type, None))
    if required_custom_section is not None:
        user_sections = [standardise_string(section_name)
                         for section_name in config.sections()]
        assert required_custom_section in user_sections, \
            '{} requires configuration section [{}] in config file'.format(
                app_type, param_parser.CUSTOM_SECTIONS[app_type])
