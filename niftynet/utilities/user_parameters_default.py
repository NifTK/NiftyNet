# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from niftynet.utilities.user_parameters_helper import float_array
from niftynet.utilities.user_parameters_helper import int_array
from niftynet.utilities.user_parameters_helper import spatialnumarray
from niftynet.utilities.user_parameters_helper import str2boolean
from niftynet.utilities.user_parameters_helper import str_array

DEFAULT_INFERENCE_OUTPUT = os.path.join('.', 'output')
DEFAULT_DATASET_SPLIT_FILE = os.path.join('.', 'dataset_split.csv')
DEFAULT_MODEL_DIR = None


def add_application_args(parser):
    # parser.add_argument(
    #     "action",
    #     help="train or inference action",
    #     choices=['train', 'inference'])

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

    return parser


def add_inference_args(parser):
    parser.add_argument(
        "--spatial_window_size",
        type=int_array,
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
        "--border",
        metavar='',
        help="[Inference only] Width of borders to crop for segmented patch",
        type=spatialnumarray,
        default=(0, 0, 0))
    return parser


def add_input_data_args(parser):
    parser.add_argument(
        "--csv_file",
        metavar='',
        type=str,
        help="Input list of subjects in csv files",
        default='')

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
        "--interp_order",
        type=int,
        choices=[0, 1, 2, 3],
        default=3,
        help="interpolation order of the input images")

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
        type=int_array,
        help="specify the spatial size of the input data (ndims <= 3)",
        default=())
    return parser


def add_network_args(parser):
    parser.add_argument(
        "--name",
        help="Choose a net from NiftyNet/niftynet/network/ or from "
             "user specified module string",
        metavar='')

    import niftynet.layer.activation
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
        "--decay",
        help="[Training only] Set weight decay",
        type=float,
        default=0)

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
        "--window_sampling",
        metavar='TYPE_STR',
        help="How to sample patches from each loaded image:"
             " 'uniform': fixed size uniformly distributed,"
             " 'resize': resize image to the patch size.",
        choices=['uniform', 'resize'],
        default='uniform')

    parser.add_argument(
        "--queue_length",
        help="Set size of preprocessing buffer queue",
        metavar='',
        type=int,
        default=5)

    import niftynet.layer.binary_masking
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
        default='')

    # TODO add choices of normalisation types
    import niftynet.utilities.histogram_standardisation as hist_std_module
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

    import niftynet.layer.binary_masking
    parser.add_argument(
        "--foreground_type",
        choices=list(
            niftynet.layer.binary_masking.SUPPORTED_MASK_TYPES),
        help="type_str of foreground masking strategy used",
        default='otsu_plus')

    parser.add_argument(
        "--normalisation",
        help="Indicates if the normalisation must be performed",
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

    try:
        from niftynet.utilities.util_import import check_module
        check_module('yaml')
        import yaml
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
    except ImportError:
        # "PyYAML module not found")
        pass

    return parser


def add_training_args(parser):
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
        "--random_flipping_axes",
        help="The axes which can be flipped to augment the data. Supply as "
             "comma-separated values within single quotes, e.g. '0,1'. Note "
             "that these are 0-indexed, so choose some combination of 0, 1.",
        type=int_array,
        default=-1)

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

    return parser
