# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import argparse
import os
import re

import niftynet.utilities.misc_csv as misc_csv
from niftynet.utilities.misc_common import look_up_operations

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from niftynet.utilities.filename_matching import KeywordsMatching

SYSTEM_SECTIONS = {'APPLICATION', 'NETWORK', 'TRAINING', 'INFERENCE'}

def _input_path_search(config):
    input_sources = {}
    # match all input modality sections (non-system sections)
    for section in config.sections():
        new_section = re.sub('[^0-9a-zA-Z]+', '_', section)
        section = __rename_section(config, section, new_section)
        if section not in SYSTEM_SECTIONS:
            items = config.items(section)
            input_sources[section] = [KeywordsMatching.from_tuple(items)]
    return input_sources

def parse_arguments_by_section(parser, section, defaults, args_from_cmd):
    section_parser = argparse.ArgumentParser(
        parents=parser,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    if section == 'application':
        section_parser = __add_application_args(section_parser)
    elif section == 'network':
        pass
    elif section == 'training' and section_parser.action == 'train':
        pass
    elif section == 'inference' and section_parser.action == 'inference':
        pass
    else:
        pass

    if defaults:
        section_parser.set_defaults(**defaults)
    section_args = section_parser.parse_args(args_from_cmd)
    return section_args

def __add_application_args(parser):
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
    return parser

def __add_network_args(parser):
    return parser

def __add_training_args(parser):
    return parser

def __add_inference_args(parser):
    return parser

def build_parser(parents, defaults):

    parser.add_argument(
        "--model_dir",
        metavar='',
        help="Directory to save/load intermediate training models and logs",
        default=os.path.join(
            os.path.dirname(__file__), '..', '..', 'models', 'model_default')
    )

    parser.add_argument(
        "--net_name",
        help="Choose a net from ./network/ ",
        metavar='',
        default='highres3dnet'
    )

    import niftynet.layer.activation
    parser.add_argument(
        "--activation_function",
        help="Specify activation function types",
        choices=list(niftynet.layer.activation.SUPPORTED_OP),
        metavar='TYPE_STR',
        default='prelu'
    )
    parser.add_argument(
        "--queue_length",
        help="Set size of preprocessing buffer queue",
        metavar='',
        type=int,
        default=20
    )
    parser.add_argument(
        "--num_threads",
        help="Set number of preprocessing threads",
        metavar='',
        type=int,
        default=2
    )
    parser.add_argument(
        "--spatial_rank",
        metavar='',
        help="Set input spatial rank",
        choices=[2, 2.5, 3],
        type=float,
        default=3
    )
    parser.add_argument(
        "--batch_size",
        metavar='',
        help="Set batch size of the net",
        type=int,
        default=20
    )
    parser.add_argument(
        "--image_size",
        metavar='',
        help="Set input image size",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--label_size",
        metavar='',
        help="Set label size of the net",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--w_map_size",
        metavar='',
        help="Set weight map size of the net",
        type=int,
        default=-1  # this option will often be unused. -1 will crash if it used and unset
    )
    parser.add_argument(
        "--num_classes",
        metavar='',
        help="Set number of classes",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--volume_padding_size",
        metavar='',
        help="Set padding size of each volume (in all dimensions)",
        type=int,
        default=5
    )
    parser.add_argument(
        "--histogram_ref_file",
        help="A reference file of histogram for intensity normalisation",
        default=os.path.join(os.path.dirname
                             (__file__), '..', 'model_ckpts', 'hist.txt')
    )
    parser.add_argument(
        "--normalisation",
        help="Indicates if the normalisation must be performed",
        default='True'
    )
    parser.add_argument(
        "--whitening",
        help="Indicates if the whitening of the data should be applied",
        default='True'
    )
    parser.add_argument(
        "--image_interp_order",
        help="image interpolation order when do resampling/rotation",
        type=int,
        default=3
    )
    parser.add_argument(
        "--label_interp_order",
        help="label interpolation order when do resampling/rotation",
        type=int,
        default=0
    )
    parser.add_argument(
        "--w_map_interp_order",
        help="weight map interpolation order when do resampling/rotation",
        type=int,
        default=3
    )
    parser.add_argument(
        "--random_flip",
        help="Indicates whether 'flipping' should be performed "
             "as a data-augmentation step. Please set --flip_axes"
             " as well for correct functioning.",
        default='False'
    )
    parser.add_argument(
        "--flip_axes",
        help="The axes which can be flipped to augment the data. Supply as "
             "comma-separated values within single quotes, e.g. '0,1'. Note "
             "that these are 0-indexed, so choose some combination of 0, 1, 2.",
        type=str,
        default=''
    )
    parser.add_argument(
        "--spatial_scaling",
        help="Indicates if the spatial scaling must be performed (zooming"
             " as an augmentation step)",
        default='False'
    )
    parser.add_argument(
        "--min_percentage",
        help="the spatial scaling factor in [min_percentage, max_percentage]",
        type=float,
        default=-10
    )
    parser.add_argument(
        "--max_percentage",
        help="the spatial scaling factor in [min_percentage, max_percentage]",
        type=float,
        default=10
    )
    parser.add_argument(
        "--reorientation",
        help="Indicates if the loaded images are put by default in the RAS "
             "orientation",
        default='False'
    )
    parser.add_argument(
        "--resampling",
        help="Indicates if the volumes must be resampled to an isotropic "
             "resolution of 1mm x 1mm x 1mm",
        default='False'
    )
    # TODO add choices of normalisation types
    parser.add_argument(
        "--norm_type",
        help="Type of normalisation to perform",
        default='percentile',
    )
    parser.add_argument(
        "--cutoff_min",
        help="Cutoff values for the normalisation process",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--cutoff_max",
        help="Cutoff values for the normalisation process",
        type=float,
        default=0.99
    )
    parser.add_argument(
        "--multimod_mask_type",
        choices=['and', 'or', 'multi'],
        help="Way of associating the masks obtained for the different "
             "modalities. 'and is the intersection, 'or' is the union "
             "and 'multi' permits each modality to use its own mask.",
        default='and'
    )
    parser.add_argument(
        "--mask_type",
        choices=['otsu_plus', 'otsu_minus',
                 'threshold_plus', 'threshold_minus', 'mean'],
        help="type of masking strategy used",
        default='otsu_plus'
    )
    parser.add_argument(
        "--rotation",
        help="Indicates if a rotation should be applied to the volume",
        default='False'
    )
    parser.add_argument(
        "--min_angle",
        help="Lower bound on the angle of rotation when rotation "
             "augmentation is enabled",
        type=float,
        default=-10.0
    )
    parser.add_argument(
        "--max_angle",
        help="Upper bound on the angle of rotation when rotation "
             "augmentation is enabled",
        type=float,
        default=10.0
    )
    parser.add_argument(
        "--num_gpus",
        help="[Training only] Set number of GPUs",
        metavar='',
        type=int,
        default=1
    )
    parser.add_argument(
        "--sample_per_volume",
        help="[Training only] Set number of samples to take from "
             "each image that was loaded in a given training epoch",
        metavar='',
        type=int,
        default=10
    )
    parser.add_argument(
        "--lr",
        help="[Training only] Set learning rate",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--decay",
        help="[Training only] Set weight decay",
        type=float,
        default=0
    )
    parser.add_argument(
        "--loss_type",
        metavar='TYPE_STR',
        help="[Training only] Specify loss type",
        default='Dice'
    )
    parser.add_argument(
        "--reg_type",
        metavar='TYPE_STR',
        help="[Training only] Specify regulariser type",
        default='L2'
    )
    parser.add_argument(
        "--starting_iter",
        metavar='', help="[Training only] Resume from iteration n",
        type=int,
        default=0
    )
    parser.add_argument(
        "--save_every_n",
        metavar='',
        help="[Training only] Model saving frequency",
        type=int,
        default=500
    )
    parser.add_argument(
        "--max_iter",
        metavar='',
        help="[Training only] Total number of iterations",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--border",
        metavar='',
        help="[Inference only] Width of borders to crop for segmented patch",
        type=int,
        default=5
    )
    parser.add_argument(
        "--inference_iter",
        metavar='',
        help="[Inference only] Use the checkpoint at this iteration for "
             "inference",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--save_seg_dir",
        metavar='',
        help="[Inference only] Prediction directory name",  # without '/'
        default=os.path.join(os.path.dirname(__file__), '..', 'model_ckpts', 'inferences')
    )
    parser.add_argument(
        "--output_interp_order",
        metavar='',
        help="[Inference only] interpolation order of the network output",
        type=int,
        default=0
    )
    parser.add_argument(
        "--output_prob",
        metavar='',
        help="[Inference only] whether to output multi-class probabilities",
        default='False'
    )
    parser.add_argument(
        "--window_sampling",
        metavar='TYPE_STR',
        help="How to sample patches from each loaded image: fixed size uniformly distributed "
             " ('uniform'), fixed size preferentially selected patches to have features "
             "of interest ('selective'), or taking the whole image and rescaling it to the "
             " patch size ('resize'). 'selective' has properties that depend"
             "on the 'min_sampling_ratio' and the 'min_numb_labels' parameters",
        choices=['uniform', 'selective','resize'],
        default='uniform'
    )
    parser.add_argument(
        "--min_sampling_ratio",
        help="Minimum ratio to satisfy in the sampling of different labels",
        default=0.00001
    )
    parser.add_argument(
        "--min_numb_labels",
        help="Minimum number of different labels present in a patch",
        default=2
    )
    parser.add_argument(
        "--max_checkpoints",
        help="Maximum number of model checkpoints that will be saved",
        type=int,
        default=10000
    )

    # will re-write defaults with config files.
    return section_args


def run():
    meta_parser = argparse.ArgumentParser(add_help=False)
    meta_parser.add_argument("-action",
                             help="train or inference",
                             choices=['train', 'inference'])
    meta_parser.add_argument("-c", "--conf",
                             help="Specify configurations from a file",
                             metavar="File", )
    meta_args, args_from_cmd = meta_parser.parse_known_args()
    if meta_args.conf and os.path.exists(meta_args.conf):
        config = configparser.ConfigParser()
        config.read([meta_args.conf])
        # initialise search of image modality filenames
        input_sources = _input_path_search(config)
    else:
        raise IOError("Configuration file not found {}".format(meta_args.conf))

    all_args = {}
    for section in config.sections():
        if section not in SYSTEM_SECTIONS:
            continue
        section_defaults = dict(config.items(section))
        import pdb; pdb.set_trace()
        section_args = parse_arguments_by_section(
                [meta_parser], section, section_defaults, args_from_cmd)
        all_args[section] = section_args
    all_args['config_file'] = meta_args.conf  # update conf path
    import pdb; pdb.set_trace()

    csv_dict = {}
    for (input_name, input_file_matcher) in input_sources.items():
        file_csv_path = os.path.join(
                meta_argss.model_dir, '{}{}'.format(input_name, '.csv'))
        misc_csv.write_matched_filenames_to_csv(
                input_file_matcher, file_csv_path)
        csv_dict[input_name] = file_csv_path
    import pdb; pdb.set_trace()
    all_args = correct_args_types(all_args)
    return all_args, csv_dict


def correct_args_types(args):
    args.reorientation = True if args.reorientation == "True" else False
    args.resampling = True if args.resampling == "True" else False
    args.normalisation = True if args.normalisation == "True" else False
    args.whitening = True if args.whitening == "True" else False
    args.rotation = True if args.rotation == "True" else False
    args.spatial_scaling = True if args.spatial_scaling == "True" else False
    args.output_prob = True if args.output_prob == "True" else False
    args.random_flip = True if args.random_flip == "True" else False
    args.flip_axes = ([int(x.strip()) for x in args.flip_axes.split(',')
                       if x.strip() in ['0', '1', '2']])
    return args


#def _eval_path_search(config):
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
#def run_eval():
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

