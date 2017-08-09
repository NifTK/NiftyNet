# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from niftynet.utilities.misc_common import look_up_operations
from niftynet.utilities.user_parameters_helper import *

SUPPORTED_TASKS = {'SEGMENTATION'}


#######################################################################
# To support a CUSTOM_SECTION in config file:
# 1) update niftynet.utilities.user_parameters_custom.SUPPORTED_TASKS
# with a CUSTOM_SECTION, this should be standardised string.
# Standardised string is defined in
# niftynet.utilities.user_parameters_helper.standardise_string

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

    from niftynet.application.segmentation_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser
