# -*- coding: utf-8 -*-
"""
Parse user configuration file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import textwrap

from niftynet.engine.application_factory import ApplicationFactory
from niftynet.engine.application_factory import SUPPORTED_APP
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig
from niftynet.utilities.user_parameters_custom import add_customised_args
from niftynet.utilities.user_parameters_default import add_application_args
from niftynet.utilities.user_parameters_default import add_inference_args
from niftynet.utilities.user_parameters_default import add_input_data_args
from niftynet.utilities.user_parameters_default import add_network_args
from niftynet.utilities.user_parameters_default import add_training_args
from niftynet.utilities.user_parameters_helper import has_section_in_config
from niftynet.utilities.user_parameters_helper import standardise_section_name
from niftynet.utilities.util_common import \
    damerau_levenshtein_distance as edit_distance
from niftynet.utilities.versioning import get_niftynet_version_string

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

SYSTEM_SECTIONS = {'SYSTEM', 'NETWORK', 'TRAINING', 'INFERENCE'}
epilog_string = \
    '\n\n======\nFor more information please visit:\n' \
    'https://github.com/NifTK/NiftyNet/tree/dev/config/README.md\n' \
    '======\n\n'


def run():
    """
    meta_parser is first used to find out location
    of the configuration file. based on the application_name
    or meta_parser.prog name, the section parsers are organised
    to find system parameters and application specific
    parameters.

    :return: system parameters is a group of parameters including
        SYSTEM_SECTIONS and app_module.REQUIRED_CONFIG_SECTION
        input_data_args is a group of input data sources to be
        used by niftynet.io.ImageReader
    """
    meta_parser = argparse.ArgumentParser(
        description="Launch a NiftyNet application.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(epilog_string))
    version_string = get_niftynet_version_string()
    meta_parser.add_argument("action",
                             help="train networks or run inferences",
                             metavar='ACTION',
                             choices=['train', 'inference'])
    meta_parser.add_argument("-v", "--version",
                             action='version',
                             version=version_string)
    meta_parser.add_argument("-c", "--conf",
                             help="specify configurations from a file",
                             metavar="CONFIG_FILE")
    meta_parser.add_argument("-a", "--application_name",
                             help="specify an application module name",
                             metavar='APPLICATION_NAME',
                             default="")

    meta_args, args_from_cmdline = meta_parser.parse_known_args()
    print(version_string)

    # read configurations, to be parsed by sections
    if not meta_args.conf:
        print("\nNo configuration file has been provided, did you "
              "forget '-c' command argument?{}".format(epilog_string))
        raise IOError

    # Resolve relative configuration file location
    config_path = os.path.expanduser(meta_args.conf)
    if not os.path.isfile(config_path):
        relative_conf_file = os.path.join(
            NiftyNetGlobalConfig().get_default_examples_folder(),
            config_path,
            config_path + "_config.ini")
        if os.path.isfile(relative_conf_file):
            config_path = relative_conf_file
            os.chdir(os.path.dirname(config_path))
        else:
            print("\nConfiguration file not found: {}.{}".format(
                config_path, epilog_string))
            raise IOError

    config = configparser.ConfigParser()
    config.read([config_path])
    app_module = None
    module_name = None
    try:
        if meta_parser.prog[:-3] in SUPPORTED_APP:
            module_name = meta_parser.prog[:-3]
        elif meta_parser.prog in SUPPORTED_APP:
            module_name = meta_parser.prog
        else:
            module_name = meta_args.application_name
        app_module = ApplicationFactory.create(module_name)
        assert app_module.REQUIRED_CONFIG_SECTION, \
            "\nREQUIRED_CONFIG_SECTION should be static variable " \
            "in {}".format(app_module)
        has_section_in_config(config, app_module.REQUIRED_CONFIG_SECTION)
    except ValueError:
        if app_module:
            section_name = app_module.REQUIRED_CONFIG_SECTION
            print('\n{} requires [{}] section in the config file.{}'.format(
                module_name, section_name, epilog_string))
        if not module_name:
            print("\nUnknown application {}, or did you forget '-a' "
                  "command argument?{}".format(module_name, epilog_string))
        raise

    # check keywords in configuration file
    check_keywords(config)

    # using configuration as default, and parsing all command line arguments
    all_args = {}
    for section in config.sections():
        # try to rename user-specified sections for consistency
        section = standardise_section_name(config, section)
        section_defaults = dict(config.items(section))
        section_args, args_from_cmdline = \
            _parse_arguments_by_section([],
                                        section,
                                        section_defaults,
                                        args_from_cmdline,
                                        app_module.REQUIRED_CONFIG_SECTION)
        all_args[section] = section_args
    # command line parameters should be valid
    assert not args_from_cmdline, \
        '\nUnknown parameter: {}{}'.format(args_from_cmdline, epilog_string)

    # split parsed results in all_args
    # into dictionaries of system_args and input_data_args
    system_args = {}
    input_data_args = {}

    # copy system default sections to ``system_args``
    for section in all_args:
        if section in SYSTEM_SECTIONS:
            system_args[section] = all_args[section]
        elif section == app_module.REQUIRED_CONFIG_SECTION:
            system_args['CUSTOM'] = all_args[section]
            vars(system_args['CUSTOM'])['name'] = module_name

    if all_args['SYSTEM'].model_dir is None:
        all_args['SYSTEM'].model_dir = os.path.join(
            os.path.dirname(meta_args.conf), 'model')

    # copy non-default sections to ``input_data_args``
    for section in all_args:
        if section in SYSTEM_SECTIONS:
            continue
        if section == app_module.REQUIRED_CONFIG_SECTION:
            continue
        input_data_args[section] = all_args[section]
        # set the output path of csv list if not exists
        csv_path = input_data_args[section].csv_file
        if os.path.isfile(csv_path):
            # don't search files if csv specified in config
            try:
                delattr(input_data_args[section], 'path_to_search')
            except AttributeError:
                pass
        else:
            input_data_args[section].csv_file = ''

    # preserve ``config_file`` and ``action parameter`` from the meta_args
    system_args['CONFIG_FILE'] = argparse.Namespace(path=meta_args.conf)
    system_args['SYSTEM'].action = meta_args.action
    return system_args, input_data_args


def _parse_arguments_by_section(parents,
                                section,
                                args_from_config_file,
                                args_from_cmd,
                                required_section):
    """
    This function first adds parameter names to a parser,
    according to the section name.
    Then it loads values from configuration files as tentative params.
    Finally it overrides existing pairs of 'name, value' with commandline
    inputs.

    Commandline inputs only override system/custom parameters.
    input data related parameters needs to be defined in config file.

    :param parents: a list, parsers will be created as
        subparsers of parents
    :param section: section name to be parsed
    :param args_from_config_file: loaded parameters from config file
    :param args_from_cmd: dictionary commandline parameters
    :return: parsed parameters of the section and unknown
        commandline params.
    """
    section_parser = argparse.ArgumentParser(
        parents=parents,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    if section == 'SYSTEM':
        section_parser = add_application_args(section_parser)
    elif section == 'NETWORK':
        section_parser = add_network_args(section_parser)
    elif section == 'TRAINING':
        section_parser = add_training_args(section_parser)
    elif section == 'INFERENCE':
        section_parser = add_inference_args(section_parser)
    elif section == required_section:
        section_parser = add_customised_args(section_parser, section.upper())
    else:
        section_parser = add_input_data_args(section_parser)
    # loading all parameters a config file first
    if args_from_config_file is not None:
        section_parser.set_defaults(**args_from_config_file)
    # input command line input overrides config file
    if (section in SYSTEM_SECTIONS) or (section == required_section):
        section_args, unknown = section_parser.parse_known_args(args_from_cmd)
        return section_args, unknown
    # don't parse user cmd for input source sections
    section_args, _ = section_parser.parse_known_args([])
    return section_args, args_from_cmd


def check_keywords(config):
    """
    check config files, validate keywords provided against
    parsers' argument list
    """
    validation_parser = argparse.ArgumentParser(
        parents=[],
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        conflict_handler='resolve')
    config_keywords = []
    for section in config.sections():
        validation_parser = add_application_args(validation_parser)
        validation_parser = add_network_args(validation_parser)
        validation_parser = add_training_args(validation_parser)
        validation_parser = add_inference_args(validation_parser)
        validation_parser = add_input_data_args(validation_parser)
        try:
            validation_parser = add_customised_args(
                validation_parser, section.upper())
        except (argparse.ArgumentError, NotImplementedError):
            pass

        if config.items(section):
            config_keywords.extend(list(dict(config.items(section))))

    default_keywords = []
    for action in validation_parser._actions:
        try:
            default_keywords.append(action.option_strings[0][2:])
        except (IndexError, AttributeError, ValueError):
            pass

    for config_key in config_keywords:
        if config_key in default_keywords:
            continue
        dists = {k: edit_distance(k, config_key) for k in default_keywords}
        closest = min(dists, key=dists.get)
        if dists[closest] <= 5:
            raise ValueError(
                'Unknown keywords in config file: By "{0}" '
                'did you mean "{1}"?\n "{0}" is '
                'not a valid option.{2}'.format(
                    config_key, closest, epilog_string))
        raise ValueError(
            'Unknown keywords in config file: [{}] -- all '
            ' possible choices are {}.{}'.format(
                config_key, default_keywords, epilog_string))
