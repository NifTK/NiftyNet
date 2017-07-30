# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from niftynet.utilities.user_parameters_helper import *

DEFAULT_INFERENCE_OUTPUT = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'outputs')

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'model_default')


def add_application_args(parser):
    parser.add_argument(
        "action",
        help="train or inference",
        choices=['train', 'inference'])

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


def add_inference_args(parser):
    parser.add_argument(
        "--border",
        metavar='',
        help="[Inference only] Width of borders to crop for segmented patch",
        type=numarray,
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
    return parser


def add_input_data_args(parser):
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
        type=numarray,
        help="input data size")

    parser.add_argument(
        "--interp_order",
        type=int,
        choices=[0, 1, 2, 3],
        default=3,
        help="interpolation order of the input images")

    return parser


def add_network_args(parser):
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
        type=numarray,
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
        type=numarray,
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


def add_training_args(parser):
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
        type=numarray,
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
        type=numarray,
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
        type=numarray,
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
        type=numarray,
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
