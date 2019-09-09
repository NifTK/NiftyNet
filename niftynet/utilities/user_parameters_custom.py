# -*- coding: utf-8 -*-
"""
This module defines task specific parameters
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from niftynet.utilities.user_parameters_helper import (add_input_name_args,
                                                       int_array, str2boolean)


#######################################################################
# To support a CUSTOM_SECTION in config file:
# (e.g., MYTASK; in parallel with SEGMENTATION, REGRESSION etc.)
#
# 1) update niftynet.utilities.user_parameters_custom.SUPPORTED_ARG_SECTIONS
# with a key-value pair:
# where the key should be MYTASK, a standardised string --
# Standardised string is defined in
# niftynet.utilities.user_parameters_helper.standardise_string
# the section name will be filtered with,
# re.sub('[^0-9a-zA-Z_\- ]+', '', input_string.strip())
#
# the value should be __add_mytask_args()
#
# 2) create a function __add_mytask_args() with task specific arguments
# this function should return an argparse obj
#
# 3) in the application file, specify:
# `REQUIRED_CONFIG_SECTION = "MYTASK"`
# so that the application will have access to the task specific arguments
#########################################################################


def add_customised_args(parser, task_name):
    """
    loading keywords arguments to parser by task name
    :param parser:
    :param task_name: supported choices are listed in `SUPPORTED_ARG_SECTIONS`
    :return: parser with updated actions
    """
    task_name = task_name.upper()
    if task_name in SUPPORTED_ARG_SECTIONS:
        return SUPPORTED_ARG_SECTIONS[task_name](parser)
    raise NotImplementedError


def __add_regression_args(parser):
    """
    keywords defined for regression tasks

    :param parser:
    :return:
    """
    parser.add_argument(
        "--loss_border",
        metavar='',
        help="Set the border size for the loss function to ignore",
        type=int,
        default=0)

    parser.add_argument(
        "--error_map",
        metavar='',
        help="Set whether to output the regression error maps (the maps "
             "will be stored in $model_dir/error_maps; the error maps "
             "can be used for window sampling).",
        type=str2boolean,
        default=False)

    from niftynet.application.regression_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_segmentation_args(parser):
    """
    keywords defined for segmentation tasks

    :param parser:
    :return:
    """
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
        "--softmax",
        metavar='',
        help="[Training only] whether to append a softmax layer to network "
             "output before feeding it into loss function",
        type=str2boolean,
        default=True)

    # for selective sampling only
    parser.add_argument(
        "--min_sampling_ratio",
        help="[Training only] Minimum ratio of samples in a window for "
             "selective sampler",
        metavar='',
        type=float,
        default=0)

    # for selective sampling only
    parser.add_argument(
        "--compulsory_labels",
        help="[Training only] List of labels to have in the window for "
             "selective sampling",
        metavar='',
        type=int_array,
        default=(0, 1))

    # for selective sampling only
    parser.add_argument(
        "--rand_samples",
        help="[Training only] Number of completely random samples per image "
             "when using selective sampler",
        metavar='',
        type=int,
        default=0)

    # for selective sampling only
    parser.add_argument(
        "--min_numb_labels",
        help="[Training only] Number of labels to have in the window for "
             "selective sampler",
        metavar='',
        type=int,
        default=1)

    # for selective sampling only
    parser.add_argument(
        "--proba_connect",
        help="[Training only] Number of labels to have in the window for "
             "selective sampler",
        metavar='',
        type=str2boolean,
        default=True)

    parser.add_argument(
        "--evaluation_units",
        help="Compute per-component metrics for per label or per connected "
             "component. [foreground, label, or cc]",
        choices=['foreground', 'label', 'cc'],
        default='foreground')

    #  for mixup augmentation
    parser.add_argument(
        "--do_mixup",
        help="Use the 'mixup' option.",
        type=str2boolean,
        default=False)

    #  for mixup augmentation
    parser.add_argument(
        "--mixup_alpha",
        help="The alpha value to parametrise the beta distribution "
             "(alpha, alpha). Default: 0.2.",
        type=float,
        default=0.2)

    #  for mixup augmentation
    parser.add_argument(
        "--mix_match",
        help="If true, matches bigger segmentations with "
             "smaller segmentations.",
        type=str2boolean,
        default=False)

    from niftynet.application.segmentation_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_gan_args(parser):
    """
    keywords defined for GAN

    :param parser:
    :return:
    """
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


def __add_classification_args(parser):
    """
    keywords defined for classification

    :param parser:
    :return:
    """
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

    from niftynet.application.classification_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_autoencoder_args(parser):
    """
    keywords defined for autoencoder

    :param parser:
    :return:
    """
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


def __add_registration_args(parser):
    """
    keywords defined for image registration

    :param parser:
    :return:
    """
    parser.add_argument(
        "--label_normalisation",
        metavar='',
        help="whether to map unique labels in the training set to "
             "consecutive integers (the smallest label will be  mapped to 0)",
        type=str2boolean,
        default=False)

    from niftynet.application.label_driven_registration import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


SUPPORTED_ARG_SECTIONS = {
    'REGRESSION': __add_regression_args,
    'SEGMENTATION': __add_segmentation_args,
    'CLASSIFICATION': __add_classification_args,
    'AUTOENCODER': __add_autoencoder_args,
    'GAN': __add_gan_args,
    'REGISTRATION': __add_registration_args
}
