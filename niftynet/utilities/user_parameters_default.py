# -*- coding: utf-8 -*-
"""
This module defines niftynet parameters and their defaults.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

from niftynet.engine.image_window_dataset import SMALLER_FINAL_BATCH_MODE
from niftynet.io.image_loader import SUPPORTED_LOADERS
from niftynet.io.image_sets_partitioner import SUPPORTED_PHASES
from niftynet.utilities.user_parameters_helper import (
    float_array, int_array, spatial_atleast3d, spatialnumarray, str2boolean,
    str_array)
from niftynet.utilities.util_import import require_module

DEFAULT_INFERENCE_OUTPUT = os.path.join('.', 'output')
DEFAULT_EVALUATION_OUTPUT = os.path.join('.', 'evaluation')
DEFAULT_DATASET_SPLIT_FILE = os.path.join('.', 'dataset_split.csv')
DEFAULT_HISTOGRAM_REF_FILE = os.path.join('.', 'histogram_ref_file.txt')
DEFAULT_MODEL_DIR = None
DEFAULT_EVENT_HANDLERS = ('model_saver', 'model_restorer', 'sampler_threading',
                          'apply_gradients', 'output_interpreter',
                          'console_logger', 'tensorboard_logger',
                          'performance_logger')

DEFAULT_ITERATION_GENERATOR = 'iteration_generator'


def add_application_args(parser):
    """
    Common keywords for all applications

    :param parser:
    :return:
    """
    parser.add_argument(
        "--cuda_devices",
        metavar='',
        help="Set CUDA_VISIBLE_DEVICES variable, e.g. '0,1,2,3'; "
             "leave blank to use the system default value",
        type=str,
        default='')

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
        "--dataset_split_file",
        metavar='',
        help="File assigning subjects to training/validation/inference subsets",
        default=DEFAULT_DATASET_SPLIT_FILE)

    parser.add_argument(
        "--event_handler",
        metavar='',
        help="String(s) representing event handler module(s)",
        type=str_array,
        default=DEFAULT_EVENT_HANDLERS)

    parser.add_argument(
        "--iteration_generator",
        metavar='',
        help='String representing an iteration generator class',
        type=str,
        default=DEFAULT_ITERATION_GENERATOR)

    return parser


def add_inference_args(parser):
    """
    keywords defined for inference action

    :param parser:
    :return:
    """
    parser.add_argument(
        "--spatial_window_size",
        type=spatial_atleast3d,
        help="Specify the spatial size of the input data (ndims <= 3)",
        default=())

    parser.add_argument(
        "--inference_iter",
        metavar='',
        help="[Inference only] Use the checkpoint at this iteration for "
             "inference",
        type=int,
        default=-1)

    parser.add_argument(
        "--dataset_to_infer",
        metavar='',
        help="[Inference only] which data set to compute inference for",
        choices=list(SUPPORTED_PHASES) + [''],
        default='')

    parser.add_argument(
        "--save_seg_dir",
        metavar='',
        help="[Inference only] Prediction directory name",  # without '/'
        default=DEFAULT_INFERENCE_OUTPUT)

    parser.add_argument(
        "--output_postfix",
        metavar='',
        help="[Inference only] Prediction filename postfix",
        default="_niftynet_out")

    parser.add_argument(
        "--output_interp_order",
        metavar='',
        help="[Inference only] interpolation order of the network output",
        type=int,
        default=0)

    parser.add_argument(
        "--border",
        metavar='',
        help="[Inference only] Width of borders to crop for segmented patch",
        type=spatialnumarray,
        default=(0, 0, 0))

    parser.add_argument(
        "--fill_constant",
        help="[Inference only] Output fill value "
             "used fill borders of output images.",
        type=float,
        default=0.0)

    return parser


def add_evaluation_args(parser):
    """
    keywords defined for evaluation action

    :param parser:
    :return:
    """
    parser.add_argument(
        "--evaluations",
        metavar='',
        help="[Evaluation only] List of evaluations to generate",
        default='')

    parser.add_argument(
        "--save_csv_dir",
        metavar='',
        help="[Evaluation only] Directory to save evaluation metrics",
        default=DEFAULT_EVALUATION_OUTPUT)

    return parser


def add_input_data_args(parser):
    """
    keywords defined for input data specification section

    :param parser:
    :return:
    """
    parser.add_argument(
        "--csv_file",
        metavar='',
        type=str,
        help="Input list of subjects in csv files",
        default='')

    parser.add_argument(
        "--csv_data_file",
        metavar='',
        type=str,
        help="Path to a csv with data; labels, features or coordinates for"
             "the patch based sampler",
        default='')

    parser.add_argument(
        "--to_ohe",
        help="Indicates if the data provided in the csv should be "
             "one-hot-encoded."
             "This is only valid when the csv_data_file has 2 columns",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--path_to_search",
        metavar='',
        type=str,
        help="Input data folder to find a list of input image files",
        default='')

    parser.add_argument(
        "--filename_contains",
        metavar='',
        type=str_array,
        help="keywords in input file names, matched filenames will be used.")

    parser.add_argument(
        "--filename_not_contains",
        metavar='',
        type=str_array,
        help="keywords in input file names, negatively matches filenames",
        default='')
    parser.add_argument(
        "--filename_removefromid",
        metavar='',
        type=str,
        help="Regular expression for extracting subject id from filename, "
             "matched pattern will be removed from the file names "
             "to form the subject id",
        default='')

    parser.add_argument(
        "--interp_order",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="interpolation order of the input images")

    parser.add_argument(
        "--loader",
        type=str,
        choices=list(SUPPORTED_LOADERS),
        default=None,
        help="Image loader to use from {}. "
             "Leave blank to try all loaders.".format(list(SUPPORTED_LOADERS)))

    parser.add_argument(
        "--pixdim",
        type=float_array,
        default=(),
        help="voxel width along each dimension")

    parser.add_argument(
        "--axcodes",
        type=str_array,
        default=(),
        help="labels for positive end of voxel axes, possible labels are"
             " ('L','R'),('P','A'),('I','S')"
             " *see also nibabel.orientations.ornt2axcodes")

    parser.add_argument(
        "--spatial_window_size",
        type=spatial_atleast3d,
        help="specify the spatial size of the input data (ndims <= 3)",
        default=())
    return parser


def add_network_args(parser):
    """
    keywords defined for network specification

    :param parser:
    :return:
    """
    import niftynet.layer.binary_masking
    import niftynet.layer.activation
    import niftynet.utilities.histogram_standardisation as hist_std_module

    parser.add_argument(
        "--name",
        help="Choose a net from NiftyNet/niftynet/network/ or from "
             "user specified module string",
        metavar='')

    parser.add_argument(
        "--activation_function",
        help="Specify activation function types",
        choices=list(niftynet.layer.activation.SUPPORTED_OP),
        metavar='TYPE_STR',
        default='relu')

    parser.add_argument(
        "--batch_size",
        metavar='',
        help="Set batch size of the net",
        type=int,
        default=2)

    parser.add_argument(
        "--smaller_final_batch_mode",
        metavar='TYPE_STR',
        help="If True, allow the final batch to be smaller "
             "if there are insufficient items left in the queue, "
             "and the batch size will be undetermined during "
             "graph construction.",
        choices=list(SMALLER_FINAL_BATCH_MODE),
        default='pad')

    parser.add_argument(
        "--decay",
        help="[Training only] Set weight decay",
        type=float,
        default=0.0)

    parser.add_argument(
        "--reg_type",
        metavar='TYPE_STR',
        help="[Training only] Specify regulariser type_str",
        type=str,
        default='L2')

    parser.add_argument(
        "--volume_padding_size",
        metavar='',
        help="Set padding size of each volume (in all dimensions)",
        type=spatialnumarray,
        default=(0, 0, 0))

    parser.add_argument(
        "--volume_padding_mode",
        metavar='',
        help="Set which type of numpy padding to do, see "
             "https://docs.scipy.org/doc/numpy-1.14.0/"
             "reference/generated/numpy.pad.html "
             "for details",
        type=str,
        default='minimum')

    parser.add_argument(
        "--volume_padding_to_size",
        help="Choose size to pad all input volumes to. Any dimensions "
             "that exceed the desired size will be kept the same. Default: "
             "(0, ) which indicates not to use this mode. ",
        type=spatialnumarray,
        default=(0,)
    )

    parser.add_argument(
        "--window_sampling",
        metavar='TYPE_STR',
        help="How to sample patches from each loaded image:"
             " 'uniform': fixed size uniformly distributed,"
             " 'resize': resize image to the patch size.",
        choices=['uniform', 'resize', 'balanced', 'weighted', 'patch'],
        default='uniform')

    parser.add_argument(
        "--force_output_identity_resizing",
        metavar=str2boolean,
        help="Forces the shape of the inferred output to match the "
        "input label shape rather than be resized to input image shape.",
        default=False)

    parser.add_argument(
        "--queue_length",
        help="Set size of preprocessing buffer queue",
        metavar='',
        type=int,
        default=5)

    parser.add_argument(
        "--multimod_foreground_type",
        choices=list(
            niftynet.layer.binary_masking.SUPPORTED_MULTIMOD_MASK_TYPES),
        help="Way of combining the foreground masks from different "
             "modalities. 'and' is the intersection, 'or' is the union "
             "and 'multi' permits each modality to use its own mask.",
        default='and')

    parser.add_argument(
        "--histogram_ref_file",
        metavar='',
        type=str,
        help="A reference file of histogram for intensity normalisation",
        default=DEFAULT_HISTOGRAM_REF_FILE)

    parser.add_argument(
        "--norm_type",
        help="Type of normalisation to perform",
        type=str,
        default='percentile',
        choices=list(hist_std_module.SUPPORTED_CUTPOINTS))

    parser.add_argument(
        "--cutoff",
        help="Cutoff values for the normalisation process",
        type=float_array,
        default=(0.01, 0.99))

    parser.add_argument(
        "--foreground_type",
        choices=list(niftynet.layer.binary_masking.SUPPORTED_MASK_TYPES),
        help="type_str of foreground masking strategy used",
        default='otsu_plus')

    parser.add_argument(
        "--normalisation",
        help="Indicates if the normalisation must be performed",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--rgb_normalisation",
        help="Indicates if RGB histogram equilisation should be performed",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--whitening",
        help="Indicates if the whitening of the data should be applied",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--normalise_foreground_only",
        help="Indicates whether a foreground mask should be applied when"
             " normalising volumes",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--weight_initializer",
        help="Set the initializer for the weight parameters",
        type=str,
        default='he_normal')

    parser.add_argument(
        "--bias_initializer",
        help="Set the initializer for the bias parameters",
        type=str,
        default='zeros')

    parser.add_argument(
        "--keep_prob",
        help="Probability that each element is kept "
             "if dropout is supported by the network",
        type=float,
        default=1.0)

    yaml = require_module('yaml', mandatory=False)
    if yaml:
        parser.add_argument(
            "--weight_initializer_args",
            help="Pass arguments to the initializer for the weight parameters",
            type=yaml.load,
            default={})
        parser.add_argument(
            "--bias_initializer_args",
            help="Pass arguments to the initializer for the bias parameters",
            type=yaml.load,
            default={})

    return parser


def add_training_args(parser):
    """
    keywords defined for the training action

    :param parser:
    :return:
    """
    parser.add_argument(
        "--optimiser",
        help="Choose an optimiser for computing graph gradients and applying",
        type=str,
        default='adam')

    parser.add_argument(
        "--sample_per_volume",
        help="[Training only] Set number of samples to take from "
             "each image that was loaded in a given training epoch",
        metavar='',
        type=int,
        default=1)

    parser.add_argument(
        "--rotation_angle",
        help="The min/max angles of rotation when rotation "
             "augmentation is enabled",
        type=float_array,
        default=())

    parser.add_argument(
        "--rotation_angle_x",
        help="The min/max angles of the x rotation when rotation "
             "augmentation is enabled",
        type=float_array,
        default=())

    parser.add_argument(
        "--rotation_angle_y",
        help="The min/max angles of the y rotation when rotation "
             "augmentation is enabled",
        type=float_array,
        default=())

    parser.add_argument(
        "--rotation_angle_z",
        help="The min/max angles of the z rotation when rotation "
             "augmentation is enabled",
        type=float_array,
        default=())

    parser.add_argument(
        "--scaling_percentage",
        help="The spatial scaling factor in [min_percentage, max_percentage]",
        type=float_array,
        default=())

    parser.add_argument(
        "--isotropic_scaling",
        help="Indicates if the same random scaling factor should be applied "
             "to each dimension",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--antialiasing",
        help="Indicates if antialiasing must be performed "
             "when randomly scaling the input images",
        type=str2boolean,
        default=True)

    parser.add_argument(
        "--bias_field_range",
        help="[Training only] The range of bias field coeffs in [min_coeff, "
             "max_coeff]",
        type=float_array,
        default=())

    parser.add_argument(
        "--bf_order",
        help="[Training only] maximal polynomial order to use for the "
             "creation of the bias field augmentation",
        metavar='',
        type=int,
        default=3)

    parser.add_argument(
        "--random_flipping_axes",
        help="The axes which can be flipped to augment the data. Supply as "
             "comma-separated values within single quotes, e.g. '0,1'. Note "
             "that these are 0-indexed, so choose some combination of 0, 1.",
        type=int_array,
        default=-1)

    # elastic deformation
    parser.add_argument(
        "--do_elastic_deformation",
        help="Enables elastic deformation",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--num_ctrl_points",
        help="Number of control points for the elastic deformation",
        type=int,
        default=4)

    parser.add_argument(
        "--deformation_sigma",
        help="The standard deviation for elastic deformation.",
        type=float,
        default=15)

    parser.add_argument(
        "--proportion_to_deform",
        help="What fraction of samples to deform elastically.",
        type=float,
        default=0.5)

    parser.add_argument(
        "--lr",
        help="[Training only] Set learning rate",
        type=float,
        default=0.01)

    parser.add_argument(
        "--loss_type",
        metavar='TYPE_STR',
        help="[Training only] Specify loss type_str",
        default='Dice')

    parser.add_argument(
        "--starting_iter",
        metavar='',
        help="[Training only] Resume from iteration n",
        type=int,
        default=0)

    parser.add_argument(
        "--save_every_n",
        metavar='',
        help="[Training only] Model saving frequency",
        type=int,
        default=500)

    parser.add_argument(
        "--tensorboard_every_n",
        metavar='',
        help="[Training only] Tensorboard summary frequency",
        type=int,
        default=20)

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

    parser.add_argument(
        "--validation_every_n",
        help="Validate every n iterations",
        type=int,
        default=-1)

    parser.add_argument(
        "--validation_max_iter",
        help="Number of validation batches to run",
        type=int,
        default=1)

    parser.add_argument(
        "--exclude_fraction_for_validation",
        help="Fraction of dataset to use for validation",
        type=float,
        default=0.)

    parser.add_argument(
        "--exclude_fraction_for_inference",
        help="Fraction of dataset to use for inference",
        type=float,
        default=0.)

    parser.add_argument(
        "--vars_to_restore",
        help="regex strings matching variable names to restore",
        type=str,
        default='')

    parser.add_argument(
        "--vars_to_freeze",
        help="regex strings matching variable to be fixed during training",
        type=str,
        default='')

    parser.add_argument(
        "--patience",
        metavar='',
        help='Number of iterations to wait before starting '
             'performance monitoring',
        type=int,
        default=100)

    parser.add_argument(
        "--early_stopping_mode",
        metavar='',
        help="Choose between {'mean', 'robust_mean', 'median', "
             "'generalisation_loss', 'median_smoothing', 'validation_up'}",
        type=str,
        default='mean')

    return parser


SUPPORTED_DEFAULT_SECTIONS = {
    'SYSTEM': add_application_args,
    'NETWORK': add_network_args,
    'TRAINING': add_training_args,
    'INFERENCE': add_inference_args,
    'EVALUATION': add_evaluation_args,
}
