# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import niftynet.utilities.util_csv as misc_csv
from niftynet.utilities.filename_matching import KeywordsMatching
from niftynet.utilities.user_parameters_custom import *
from niftynet.utilities.user_parameters_default import *

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

# sections not in SYSTEM_SECTIONS or CUSTOM_SECTIONS will be
# treated as input data source specifications
SYSTEM_SECTIONS = {'APPLICATION', 'NETWORK', 'TRAINING', 'INFERENCE'}
CUSTOM_SECTIONS = {'net_segment.py': 'SEGMENTATION',
                   'net_gan.py': 'GAN',
                   'net_autoencoder.py': 'AUTOENCODER'}


def run():
    # meta_parser: to find out location of the configuration file
    meta_parser = argparse.ArgumentParser(add_help=False)
    meta_parser.add_argument("-c", "--conf",
                             help="Specify configurations from a file",
                             metavar="File", )
    meta_args, args_from_cmd = meta_parser.parse_known_args()

    # read configurations, to be parsed by sections
    if (meta_args.conf is None) or (not os.path.isfile(meta_args.conf)):
        raise IOError(
            "Configuration file not found {}".format(meta_args.conf))
    config = configparser.ConfigParser()
    config.read([meta_args.conf])
    try:
        required_section_name = look_up_operations(meta_parser.prog,
                                                   CUSTOM_SECTIONS)
        search_section_in_config(config, required_section_name)
    except ValueError:
        raise ValueError(
            '{} requires [{}] section in the config file'.format(
                meta_parser.prog, CUSTOM_SECTIONS.get(meta_parser.prog, None)))

    # using configuration as default, and parsing all command line arguments
    args_remaining = args_from_cmd
    all_args = {}
    for section in config.sections():
        # try to rename user-specified sections for consistency
        section = standardise_section_name(config, section)
        section_defaults = dict(config.items(section))
        section_args, args_remaining = _parse_arguments_by_section(
            [], section, section_defaults, args_remaining)
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
        elif section in set(CUSTOM_SECTIONS.values()):
            system_args['CUSTOM'] = all_args[section]
            vars(system_args['CUSTOM'])['name'] = meta_parser.prog
        else:
            input_data_args[section] = _make_csv_files(all_args, section)
    # update conf path
    system_args['CONFIG_FILE'] = argparse.Namespace(path=meta_args.conf)
    return system_args, input_data_args


def _parse_arguments_by_section(
        parents, section, args_from_config_file, args_from_cmd):
    """
    This function first add parameter names to a parser,
    according to the section name.
    Then it loads values from configuration files as tentative params.
    Finally it overrides existing pairs of 'name, value' with commandline
    inputs.

    Commandline inputs only overrides system/custom parameters.
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
    elif section in set(CUSTOM_SECTIONS.values()):
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


def _make_csv_files(all_args, section):
    #  set section_args.csv_file to a valid csv file
    section_args = all_args[section]
    input_csv = section_args.csv_file
    if (input_csv is None) or (not os.path.isfile(input_csv)):
        input_csv = os.path.join(all_args['APPLICATION'].model_dir,
                                 '{}{}'.format(section, '.csv'))
    # write a new csv file if it doesn't exist
    if not os.path.isfile(input_csv):
        print('search file folders ignored, '
                        'writing csv file {}'.format(input_csv))
        section_tuple = section_args.__dict__.items()
        matcher = KeywordsMatching.from_tuple(section_tuple)
        misc_csv.match_and_write_filenames_to_csv([matcher], input_csv)
    else:
        print('using existing csv file {}, '
                        'file folder parameters ignored'.format(input_csv))
    section_args.csv_file = input_csv
    if not os.path.isfile(section_args.csv_file):
        raise IOError(
            "unable to find/create list of input filenames"
            "as a csv file {} for config"
            "section [{}]".format(section_args.csv_file, section))
    return section_args

# def _eval_path_search(config):
#    # match flexible input modality sections
#    output_keywords = []
#    ref_keywords = []
#    data_keywords = []
#    for section in config.sections():
#        section = section.lower()
#        if 'output' in section:
#            output_keywords.append(config.items(section))
#        elif 'ref' in section:
#            ref_keywords.append(config.items(section))
#        elif 'data' in section:
#            data_keywords.append(config.items(section))
#    output_matcher = [KeywordsMatching.from_tuple(mod_info)
#                      for mod_info in output_keywords]
#    ref_matcher = [KeywordsMatching.from_tuple(mod_info)
#                   for mod_info in ref_keywords]
#    data_matcher = [KeywordsMatching.from_tuple(mod_info)
#                    for mod_info in data_keywords]
#    return output_matcher, ref_matcher, data_matcher
#
# def run_eval():
#    meta_parser = argparse.ArgumentParser(add_help=False)
#    meta_parser.add_argument("-c", "--conf",
#                             help="Specify configurations from a file",
#                             metavar="File")
#    config_file = os.path.join(os.path.dirname(__file__),
#                               '..', 'config', 'default_eval_config.txt')
#    defaults = {"conf": config_file}
#    meta_parser.set_defaults(**defaults)
#    meta_args, remaining_argv = meta_parser.parse_known_args()
#    try:
#        config = configparser.ConfigParser()
#        config.read([meta_args.conf])
#        # initialise search of image modality filenames
#        output_matcher, ref_matcher, data_matcher = _eval_path_search(
#            config)
#        defaults = dict(config.items("settings"))
#    except Exception as e:
#        raise ValueError('configuration file not found')
#
#    parser = argparse.ArgumentParser(
#        parents=[meta_parser],
#        description=__doc__,
#        formatter_class=argparse.RawDescriptionHelpFormatter)
#    parser.set_defaults(**defaults)
#    parser.add_argument("action",
#                        help="compute ROI statistics or compare segmentation maps",
#                        choices=['roi', 'compare'])
#    parser.add_argument("--threshold",
#                        help="threshold to obtain binary segmentation",
#                        type=float)
#    parser.add_argument("--step",
#                        help="step of increment when considering probabilistic segmentation",
#                        type=float)
#    parser.add_argument("--ref_dir",
#                        help="path to the image to use as reference")
#    parser.add_argument("--seg_dir",
#                        help="path where to find the images to evaluate")
#    parser.add_argument("--img_dir",
#                        help="path where to find the images to evaluate")
#    parser.add_argument("--save_csv_dir",
#                        help="path where to save the output csv file")
#    parser.add_argument("--ext",
#                        help="extension of the image files to be read")
#    parser.add_argument("--seg_type",
#                        help="type of input: discrete maps or probabilistic maps")
#    args = parser.parse_args(remaining_argv)
#    # creating output
#    image_csv_path = os.path.join(args.save_csv_dir, 'image_files.csv')
#    misc_csv.write_matched_filenames_to_csv(output_matcher, image_csv_path)
#
#    if ref_matcher:
#        ref_csv_path = os.path.join(args.save_csv_dir, 'ref_files.csv')
#        misc_csv.write_matched_filenames_to_csv(ref_matcher, ref_csv_path)
#    else:
#        ref_csv_path = None
#    if data_matcher:
#        data_csv_path = os.path.join(args.save_csv_dir, 'data_files.csv')
#        misc_csv.write_matched_filenames_to_csv(data_matcher, data_csv_path)
#    else:
#        data_csv_path = None
#    csv_dict = {'input_image_file': image_csv_path,
#                'target_image_file': ref_csv_path,
#                'weight_map_file': data_csv_path,
#                'target_note': None}
#    return args, csv_dict
