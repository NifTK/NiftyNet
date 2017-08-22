# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from niftynet.utilities.user_parameters_helper import *
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_TASKS = {'SEGMENTATION', 'GAN', 'AUTOENCODER'}


#######################################################################
# To support a CUSTOM_SECTION in config file:
# 1) update niftynet.utilities.user_parameters_custom.SUPPORTED_TASKS
# with a CUSTOM_SECTION, this should be standardised string.
# Standardised string is defined in
# niftynet.utilities.user_parameters_helper.standardise_string
# the section name will be filtered with:
#   re.sub('[^0-9a-zA-Z ]+', '', input_string.strip())

# 2) appending add_customised_args() with a function add_*_args()
# this function should return an argparse obj
# when task_name matches CUSTOM_SECTION.

# 3) update niftynet.utilities.user_parameters_parser.CUSTOM_SECTIONS
# creat a dictionary item with 'net_[task].py': CUSTOM_SECTION
#########################################################################


def add_customised_args(parser, task_name):
    task_name = look_up_operations(task_name.upper(), SUPPORTED_TASKS)
    if task_name == 'SEGMENTATION':
        return __add_segmentation_args(parser)
    elif task_name == 'GAN':
        return __add_gan_args(parser)
    elif task_name == 'AUTOENCODER':
        return __add_autoencoder_args(parser)
    else:
        raise NotImplemented


def __add_segmentation_args(parser):
    parser.add_argument(
        "--num_classes",
        metavar='',
        help="Set number of classes",
        type=int,
        default=-1)

    parser.add_argument(
        "--output_prob",
        metavar='',
        help="[Inference only] whether to output multi-class probabilities",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--label_normalisation",
        metavar='',
        help="whether to map unique labels in the training set to "
             "consecutive integers (the smallest label will be  mapped to 0)",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--min_numb_labels",
        help="Minimum number of different labels present in a patch",
        type=int_array,
        default=2)

    parser.add_argument(
        "--min_sampling_ratio",
        help="Minimum ratio to satisfy in the sampling of different labels",
        type=float,
        default=0.00001)

    from niftynet.application.segmentation_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_gan_args(parser):
    parser.add_argument(
        "--noise_size",
        metavar='',
        help="length of the noise vector",
        type=int,
        default=-1)

    parser.add_argument(
        "--n_interpolations",
        metavar='',
        help="the method of generating window from image",
        type=int,
        default=10)

    from niftynet.application.gan_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_autoencoder_args(parser):
    from niftynet.application.autoencoder_application import SUPPORTED_INFERENCE
    parser.add_argument(
        "--inference_type",
        metavar='',
        help="choose an inference type for the trained autoencoder",
        choices=list(SUPPORTED_INFERENCE))

    parser.add_argument(
        "--noise_stddev",
        metavar='',
        help="standard deviation of noise when inference type is sample",
        type=float)

    parser.add_argument(
        "--n_interpolations",
        metavar='',
        help="the method of generating window from image",
        type=int,
        default=10)

    from niftynet.application.autoencoder_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser
