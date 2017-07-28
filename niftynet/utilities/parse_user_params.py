# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import argparse
import os
import re

import niftynet.utilities.misc_csv as misc_csv
from niftynet.utilities.filename_matching import KeywordsMatching

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

SYSTEM_SECTIONS = {'APPLICATION', 'NETWORK', 'TRAINING', 'INFERENCE'}
DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'model_default')
DEFAULT_INFERENCE_OUTPUT = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'outputs')
TRUE_VALUE = {'yes', 'true', 't', 'y', '1'}
FALSE_VALUE = {'no', 'false', 'f', 'n', '0'}
ARRAY_TYPES = {"(": ")", "[": "]"}


def run():
    meta_parser = argparse.ArgumentParser(add_help=False)
    meta_parser.add_argument("-c", "--conf",
                             help="Specify configurations from a file",
                             metavar="File", )
    meta_args, args_from_cmd = meta_parser.parse_known_args()
    if meta_args.conf and os.path.exists(meta_args.conf):
        config = configparser.ConfigParser()
        config.read([meta_args.conf])
    else:
        raise IOError(
            "Configuration file not found {}".format(meta_args.conf))

    all_args = {}
    args_remaining = args_from_cmd
    for section in config.sections():
        # try to rename user-specified sections for consistency
        new_section = re.sub('[^0-9a-zA-Z]+', '_', section)
        section = __rename_section(config, section, new_section)

        section_defaults = dict(config.items(section))
        section_args, args_remaining = _parse_arguments_by_section(
            [], section, section_defaults, args_remaining)
        all_args[section] = section_args

    if not args_remaining == []:
        raise ValueError(
            'unknown parameter: {}'.format(args_remaining))

    # converting user-specified sections as input data sources
    csv_dict = {}
    for section in all_args:
        if section in SYSTEM_SECTIONS:
            continue
        section_args = all_args[section]
        input_csv = section_args.csv_file
        if input_csv is None or not os.path.isfile(input_csv):
            model_dir = all_args['APPLICATION'].model_dir
            input_csv = os.path.join(model_dir,
                                     '{}{}'.format(section, '.csv'))
            csv_dict[section] = input_csv
        # write a new csv file if it doesn't exist
        if not os.path.isfile(input_csv):
            print('writing new csv')
            section_tuple = section_args.__dict__.items()
            matcher = KeywordsMatching.from_tuple(section_tuple)
            misc_csv.match_and_write_filenames_to_csv([matcher], input_csv)

        if not os.path.isfile(input_csv):
            raise ValueError(
                    "unable to find/create list of input filenames"
                    "as a csv file {} for config"
                    "section [{}]".format(input_csv, section))
    # update conf path
    all_args['config_file'] = argparse.Namespace(path=meta_args.conf)
    return all_args, csv_dict


def _parse_arguments_by_section(parents, section, defaults, args_from_cmd):
    section_parser = argparse.ArgumentParser(
        parents=parents,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    if section == 'APPLICATION':
        section_parser = __add_application_args(section_parser)
    elif section == 'NETWORK':
        section_parser = __add_network_args(section_parser)
    elif section == 'TRAINING':
        section_parser = __add_training_args(section_parser)
    elif section == 'INFERENCE':
        section_parser = __add_inference_args(section_parser)
    else:
        section_parser = __add_data_source_args(section_parser)
    if defaults is not None:
        section_parser.set_defaults(**defaults)
    # TODO: setting multiple user-specified sections from cmd
    section_args, unknown = section_parser.parse_known_args(args_from_cmd)
    return section_args, unknown


def __add_data_source_args(parser):
    parser.add_argument(
        "--csv_file",
        metavar='',
        type=str,
        help="Input list of subjects in csv files")

    parser.add_argument(
        "--path_to_search",
        metavar='',
        type=str,
        help="Input data folder to find a list of input image files")

    parser.add_argument(
        "--filename_contains",
        metavar='',
        type=str,
        help="keywords in input file names, matched filenames will be used.")

    parser.add_argument(
        "--filename_not_contains",
        metavar='',
        type=str,
        help="keywords in input file names, negatively matches filenames")

    parser.add_argument(
        "--size",
        type=str2array,
        help="input data size")

    parser.add_argument(
        "--interp_order",
        type=int,
        choices=[0, 1, 2, 3],
        help="interpolation order of the input images")

    return parser


def __add_application_args(parser):
    parser.add_argument(
        "action",
        help="train or inference",
        choices=['train', 'inference'])

    parser.add_argument(
        "--type",
        help="Choose the type of problem you are solving",
        choices=['segmentation', 'autoencoder', 'gan'],
        default='segmentation',
        metavar='')

    parser.add_argument(
        "--cuda_devices",
        metavar='',
        help="Set CUDA_VISIBLE_DEVICES variable, e.g. '0,1,2,3'; " \
             "leave blank to use the system default value",
        default='""')

    parser.add_argument(
        "--num_threads",
        help="Set number of preprocessing threads",
        metavar='',
        type=int,
        default=2)

    parser.add_argument(
        "--num_gpus",
        help="Set number of training GPUs",
        metavar='',
        type=int,
        default=1)

    parser.add_argument(
        "--model_dir",
        metavar='',
        help="Directory to save/load intermediate training models and logs",
        default=DEFAULT_MODEL_DIR)

    parser.add_argument(
        "--queue_length",
        help="Set size of preprocessing buffer queue",
        metavar='',
        type=int,
        default=20)

    return parser


def __add_network_args(parser):
    parser.add_argument(
        "--name",
        help="Choose a net from NiftyNet/niftynet/network/",
        metavar='')

    import niftynet.layer.activation
    parser.add_argument(
        "--activation_function",
        help="Specify activation function types",
        choices=list(niftynet.layer.activation.SUPPORTED_OP),
        metavar='TYPE_STR',
        default='prelu')

    # TODO: maybe redundant
    parser.add_argument(
        "--spatial_rank",
        metavar='',
        help="Set input spatial rank",
        choices=[2, 2.5, 3],
        type=float,
        default=3)

    parser.add_argument(
        "--batch_size",
        metavar='',
        help="Set batch size of the net",
        type=int,
        default=20)

    # TODO: app specific
    parser.add_argument(
        "--num_classes",
        metavar='',
        help="Set number of classes",
        type=int,
        default=-1)

    parser.add_argument(
        "--decay",
        help="[Training only] Set weight decay",
        type=float,
        default=0)

    import niftynet.layer.loss
    parser.add_argument(
        "--reg_type",
        metavar='TYPE_STR',
        choices=list(niftynet.layer.loss.SUPPORTED_OPS),
        help="[Training only] Specify regulariser type",
        default='Dice')

    # TODO: anisotropic
    parser.add_argument(
        "--volume_padding_size",
        metavar='',
        help="Set padding size of each volume (in all dimensions)",
        type=str2array,
        default=5)

    import niftynet.layer.binary_masking
    parser.add_argument(
        "--multimod_mask_type",
        choices=list(
            niftynet.layer.binary_masking.SUPPORTED_MULTIMOD_MASK_TYPES),
        help="Way of associating the masks obtained for the different "
             "modalities. 'and is the intersection, 'or' is the union "
             "and 'multi' permits each modality to use its own mask.",
        default='and')

    parser.add_argument(
        "--histogram_ref_file",
        metavar='',
        type=str,
        help="A reference file of histogram for intensity normalisation")

    # TODO add choices of normalisation types
    parser.add_argument(
        "--norm_type",
        help="Type of normalisation to perform",
        type=str,
        default='percentile')

    parser.add_argument(
        "--cutoff",
        help="Cutoff values for the normalisation process",
        type=str2array,
        default=(0.01, 0.99))

    import niftynet.layer.binary_masking
    parser.add_argument(
        "--mask_type",
        choices=list(
            niftynet.layer.binary_masking.SUPPORTED_MASK_TYPES),
        help="type of masking strategy used",
        default='otsu_plus')

    parser.add_argument(
        "--reorientation",
        help="Indicates if the loaded images are put by default in the RAS "
             "orientation",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--resampling",
        help="Indicates if the volumes must be resampled to an isotropic "
             "resolution of 1mm x 1mm x 1mm",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--normalisation",
        help="Indicates if the normalisation must be performed",
        type=str2boolean,
        default=True)

    parser.add_argument(
        "--whitening",
        help="Indicates if the whitening of the data should be applied",
        type=str2boolean,
        default=True)

    return parser


def __add_training_args(parser):
    parser.add_argument(
        "--sample_per_volume",
        help="[Training only] Set number of samples to take from "
             "each image that was loaded in a given training epoch",
        metavar='',
        type=int,
        default=10)

    parser.add_argument(
        "--rotation",
        help="Indicates if a rotation should be applied to the volume",
        type=str2boolean,
        default=False)

    # TODO: changed parameters to tuple
    parser.add_argument(
        "--rotation_angle",
        help="The min/max angles of rotation when rotation "
             "augmentation is enabled",
        type=str2array,
        default=(-10.0, 10.0))

    parser.add_argument(
        "--spatial_scaling",
        help="Indicates if the spatial scaling must be performed (zooming"
             " as an augmentation step)",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--scaling_percentage",
        help="the spatial scaling factor in [min_percentage, max_percentage]",
        type=str2array,
        default=(-10.0, 10.0))

    parser.add_argument(
        "--window_sampling",
        metavar='TYPE_STR',
        help="How to sample patches from each loaded image:"
             " 'uniform': fixed size uniformly distributed,"
             " 'selective': selective sampling by properties of"
             "  'min_sampling_ratio' and the 'min_numb_labels' parameters"
             " 'resize': resize image to the patch size.",
        choices=['uniform', 'selective', 'resize'],
        default='uniform')

    parser.add_argument(
        "--min_numb_labels",
        help="Minimum number of different labels present in a patch",
        type=str2array,
        default=2)

    parser.add_argument(
        "--min_sampling_ratio",
        help="Minimum ratio to satisfy in the sampling of different labels",
        type=float,
        default=0.00001)

    parser.add_argument(
        "--random_flip",
        help="Indicates whether 'flipping' should be performed "
             "as a data-augmentation step. Please set --flip_axes"
             " as well for correct functioning.",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--flip_axes",
        help="The axes which can be flipped to augment the data. Supply as "
             "comma-separated values within single quotes, e.g. '0,1'. Note "
             "that these are 0-indexed, so choose some combination of 0, 1.",
        type=str2array,
        default=0)

    parser.add_argument(
        "--lr",
        help="[Training only] Set learning rate",
        type=float,
        default=0.01)

    parser.add_argument(
        "--loss_type",
        metavar='TYPE_STR',
        help="[Training only] Specify loss type",
        default='Dice')

    parser.add_argument(
        "--starting_iter",
        metavar='', help="[Training only] Resume from iteration n",
        type=int,
        default=0)

    parser.add_argument(
        "--save_every_n",
        metavar='',
        help="[Training only] Model saving frequency",
        type=int,
        default=500)

    parser.add_argument(
        "--max_iter",
        metavar='',
        help="[Training only] Total number of iterations",
        type=int,
        default=10000)

    parser.add_argument(
        "--max_checkpoints",
        help="Maximum number of model checkpoints that will be saved",
        type=int,
        default=100)

    return parser


def __add_inference_args(parser):
    parser.add_argument(
        "--border",
        metavar='',
        help="[Inference only] Width of borders to crop for segmented patch",
        type=str2array,
        default=5)

    parser.add_argument(
        "--inference_iter",
        metavar='',
        help="[Inference only] Use the checkpoint at this iteration for "
             "inference",
        type=int)

    parser.add_argument(
        "--save_seg_dir",
        metavar='',
        help="[Inference only] Prediction directory name",  # without '/'
        default=DEFAULT_INFERENCE_OUTPUT)

    parser.add_argument(
        "--output_interp_order",
        metavar='',
        help="[Inference only] interpolation order of the network output",
        type=int,
        default=0)

    parser.add_argument(
        "--output_prob",
        metavar='',
        help="[Inference only] whether to output multi-class probabilities",
        type=str2boolean,
        default=False)

    return parser


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

def __rename_section(configparser, old_name, new_name):
    """
    rename configparser section
    This helper is useful when user specifies complex section names
    """
    if old_name == new_name:
        return old_name
    items = configparser.items(old_name)
    configparser.add_section(new_name)
    for (name, value) in items:
        configparser.set(new_name, name, value)
    configparser.remove_section(old_name)
    return new_name


def str2boolean(string_input):
    if string_input.lower() in TRUE_VALUE:
        return True
    elif string_input.lower() in FALSE_VALUE:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# TODO: passing arrays to application
def str2array(string_input):
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
