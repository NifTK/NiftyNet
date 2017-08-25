# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from niftynet.engine.application_factory import ApplicationFactory
from niftynet.engine.application_factory import SUPPORTED_APP
from niftynet.utilities.user_parameters_custom import *
from niftynet.utilities.user_parameters_default import *
from niftynet.utilities.versioning import get_niftynet_version_string

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

SYSTEM_SECTIONS = {'APPLICATION', 'NETWORK', 'TRAINING', 'INFERENCE'}


def run():
    # meta_parser: to find out location of the configuration file
    meta_parser = argparse.ArgumentParser(add_help=False)
    version_string = get_niftynet_version_string()
    meta_parser.add_argument("-v", "--version",
                             action='version', version=version_string)
    meta_parser.add_argument("-c", "--conf",
                             help="Specify configurations from a file",
                             metavar="File", )
    meta_parser.add_argument("-a", "--application_name",
                             help="Specify application name",
                             default="", )
    meta_args, args_from_cmd = meta_parser.parse_known_args()
    print(version_string)

    # read configurations, to be parsed by sections
    if (meta_args.conf is None) or (not os.path.isfile(meta_args.conf)):
        raise IOError(
            "Configuration file not found {}".format(meta_args.conf))
    config = configparser.ConfigParser()
    config.read([meta_args.conf])
    try:
        if meta_parser.prog[:-3] in SUPPORTED_APP:
            module_name = meta_parser.prog[:-3]
        else:
            module_name = meta_args.application_name
        app_module = ApplicationFactory.create(module_name)
        assert app_module.REQUIRED_CONFIG_SECTION, \
            "REQUIRED_CONFIG_SECTION should be static variable " \
            "in {}".format(app_module)
        required_section_name = app_module.REQUIRED_CONFIG_SECTION
        search_section_in_config(config, required_section_name)
    except ValueError:
        raise ValueError(
            '{} requires [{}] section in the config file'.format(
                module_name, required_section_name))

    # using configuration as default, and parsing all command line arguments
    args_remaining = args_from_cmd
    all_args = {}
    for section in config.sections():
        # try to rename user-specified sections for consistency
        section = standardise_section_name(config, section)
        section_defaults = dict(config.items(section))
        section_args, args_remaining = \
            _parse_arguments_by_section([],
                                        section,
                                        section_defaults,
                                        args_remaining,
                                        required_section_name)
        all_args[section] = section_args
    if not args_remaining == []:
        raise ValueError('unknown parameter: {}'.format(args_remaining))

    # split parsed results in all_args
    # into dictionary of system_args and input_data_args
    system_args = {}
    input_data_args = {}
    for section in all_args:
        if section in SYSTEM_SECTIONS:
            system_args[section] = all_args[section]
        elif section == required_section_name:
            system_args['CUSTOM'] = all_args[section]
            vars(system_args['CUSTOM'])['name'] = module_name
        else:
            input_data_args[section] = all_args[section]
            # set the output path of csv list if not exists
            csv_path = input_data_args[section].csv_file
            if not os.path.isfile(csv_path):
                csv_filename = os.path.join(all_args['APPLICATION'].model_dir,
                                            '{}.csv'.format(section))
                input_data_args[section].csv_file = csv_filename
            else:
                # don't search files if csv specified in config
                delattr(input_data_args[section], 'path_to_search')
    # update conf path
    system_args['CONFIG_FILE'] = argparse.Namespace(path=meta_args.conf)
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

    if section == 'APPLICATION':
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
    # TODO: setting multiple user-specified sections from cmd
    # input command line input overrides config file
    section_args, unknown = section_parser.parse_known_args(args_from_cmd)
    return section_args, unknown
