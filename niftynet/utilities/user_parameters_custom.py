# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from niftynet.utilities.user_parameters_helper import add_input_name_args
from niftynet.utilities.user_parameters_helper import int_array
from niftynet.utilities.user_parameters_helper import str2boolean


#######################################################################
# To support a CUSTOM_SECTION in config file:
# (e.g., MYTASK; in parallel with SEGMENTATION, REGRESSION etc.)
#
# 1) update niftynet.utilities.user_parameters_custom.SUPPORTED_ARG_SECTIONS
# with a key-value pair:
#   where the key should be MYTASK, a standardised string --
#   Standardised string is defined in
#   niftynet.utilities.user_parameters_helper.standardise_string
#   the section name will be filtered with,
#     re.sub('[^0-9a-zA-Z ]+', '', input_string.strip())
#
#   the value should be __add_mytask_args()
#
# 2) create a function __add_mytask_args() with task specific arguments
# this function should return an argparse obj
#
# 3) in the application file, specify:
#    `REQUIRED_CONFIG_SECTION = "MYTASK"`
# so that the application will have access to the task specific arguments

#########################################################################


def add_customised_args(parser, task_name):
    task_name = task_name.upper()
    if task_name in SUPPORTED_ARG_SECTIONS:
        return SUPPORTED_ARG_SECTIONS[task_name](parser)
    else:
        raise NotImplementedError


def __add_regression_args(parser):
    parser.add_argument(
        "--loss_border",
        metavar='',
        help="Set the border size for the loss function to ignore",
        type=int,
        default=0)

    from niftynet.application.regression_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


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
        help="choose an inference type_str for the trained autoencoder",
        choices=list(SUPPORTED_INFERENCE))

    parser.add_argument(
        "--noise_stddev",
        metavar='',
        help="standard deviation of noise when inference type_str is sample",
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


SUPPORTED_ARG_SECTIONS = {
    'REGRESSION': __add_regression_args,
    'SEGMENTATION': __add_segmentation_args,
    'AUTOENCODER': __add_autoencoder_args,
    'GAN': __add_gan_args
}
