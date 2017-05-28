# -*- coding: utf-8 -*-
import os
import argparse
import configparser


def run():
    file_parser = argparse.ArgumentParser(add_help=False)
    file_parser.add_argument("-c", "--conf",
                             help="Specify configurations from a file",
                             metavar="File")
    config_file = os.path.join(os.path.dirname(__file__),
                               '../config/multimodal_config_cs.txt')
    defaults = {"conf": config_file}
    file_parser.set_defaults(**defaults)
    file_arg, remaining_argv = file_parser.parse_known_args()

    if file_arg.conf:
        config = configparser.SafeConfigParser()
        config.read([file_arg.conf])
        defaults = dict(config.items("settings"))

    parser = argparse.ArgumentParser(
        parents=[file_parser],
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.set_defaults(**defaults)

    parser.add_argument(
        "action", help="train or inference", choices=['train', 'inference'])

    parser.add_argument(
        "--cuda_devices",
        metavar='',
        help="Set CUDA_VISIBLE_DEVICES variable, e.g. '0,1,2,3'; " \
                "leave blank to use the system default value")
    parser.add_argument(
        "--model_dir",
        metavar='',
        help="Directory to save/load intermediate training models and logs")
    parser.add_argument(
        "--net_name",
        help="Choose a net from ./network/ ",
        metavar='')

    parser.add_argument(
        "--queue_length",
        help="Set size of preprocessing buffer queue",
        metavar='',
        type=int)
    parser.add_argument(
        "--num_threads",
        help="Set number of preprocessing threads",
        metavar='',
        type=int)

    parser.add_argument(
        "--batch_size", metavar='', help="Set batch size of the net", type=int)
    parser.add_argument(
        "--image_size", metavar='', help="Set input image size", type=int)
    parser.add_argument(
        "--label_size", metavar='', help="Set label size of the net", type=int)
    parser.add_argument(
        "--num_classes", metavar='', help="Set number of classes", type=int)

    parser.add_argument(
        "--volume_padding_size",
        metavar='',
        help="Set padding size of each volume (in all dimensions)",
        type=int)
    parser.add_argument(
        "--histogram_ref_file",
        help="A reference file of histogram for intensity normalisation")
    parser.add_argument(
        "--flag_normalisation",
        help="Indicates if the normalisation must be performed"
    )
    parser.add_argument(
        "--flag_whitening",
        help="Indicates if the whitening of the data should be applied"
    )
    parser.add_argument(
        "--flag_spatial_scaling",
        help="Indicates if the spatial scaling must be performed"
    )
    parser.add_argument(
        "--flag_orientation",
        help="Indicates if the loaded images are put by default in the RAS "
             "orientation"
    )
    parser.add_argument(
        "--flag_isotropic",
        help="Indicates if the volumes must be interpolated to be in "
             "representing images of 1 1 1 resolution"
    )
    parser.add_argument(
        "--flag_saving_norm",
        help="Indicates if the normalisation must be saved"
    )
    parser.add_argument(
        "--norm_type",
        default='percentile',
        help="Type of normalisation to perform"
    )
    parser.add_argument(
        "--flag_saving_mask",
        help="Indicates if generated masks must be saved or not"
    )
    parser.add_argument(
        #TODO refactor this part to allow to use values different from default
        "--norm_cutoff",
        default=[0.01, 0.99],
        nargs=2,
        help="Cutoff values for the normalisation process"
    )
    parser.add_argument(
        "--multimod_mask_type",
        choices=['and', 'or', 'multi'],
        help="Way of associating the masks obtained for the different "
             "modalities"
    )
    parser.add_argument(
        "--mask_type",
        choices=['otsu_plus','otsu_minus','val_plus','val_minus'],
        help="type of masking strategy used"
    )
    parser.add_argument(
        "--flag_rotation",
        help="Indicates if a rotation should be applied"
    )

    parser.add_argument(
        "--num_gpus",
        help="[Training only] Set number of GPUs",
        metavar='',
        type=int)
    parser.add_argument(
        "--sample_per_volume",
        help="[Training only] Set number of samples per image in each batch",
        metavar='',
        type=int)
    # TODO remove the trailing '/'
    parser.add_argument(
        "--train_data_dir",
        metavar='',
        nargs='+',
        help="[Training only] Specify training input volume directory")

    parser.add_argument(
        "--lr",
        help="[Training only] Set learning rate", type=float)
    parser.add_argument(
        "--decay",
        help="[Training only] Set weight decay", type=float)
    parser.add_argument(
        "--loss_type",
        metavar='TYPE_STR', help="[Training only] Specify loss type")
    parser.add_argument(
        "--reg_type",
        metavar='TYPE_STR', help="[Training only] Specify regulariser type")
    parser.add_argument(
        "--starting_iter",
        metavar='', help="[Training only] Resume from iteration n", type=int)
    parser.add_argument(
        "--save_every_n",
        metavar='', help="[Training only] Model saving frequency", type=int)
    parser.add_argument(
        "--max_iter",
        metavar='', help="[Training only] Total number of iterations", type=int)

    parser.add_argument(
        "--border",
        metavar='',
        help="[Inference only] Width of cropping borders for segmented patch",
        type=int)
    parser.add_argument(
        "--pred_iter",
        metavar='',
        help="[Inference only] Using model at iteration n",
        type=int)
    parser.add_argument(
        "--save_seg_dir",
        metavar='',
        help="[Inference only] Prediction directory name")  # without '/'
    parser.add_argument(
        "--eval_data_dir",
        metavar='',
        help="[Inference only] Directory of image to be segmented")  # without '/'
    parser.add_argument(
        "--min_sampling_ratio",
        help="Minumum ratio to satisfy in the sampling of different labels"
    )
    parser.add_argument(
        "--min_numb_labels",
        help="Minimum number of different labels present in a patch"
    )
    args = parser.parse_args(remaining_argv)
    return args


def run_eval():
    file_parser = argparse.ArgumentParser(add_help=False)
    file_parser.add_argument("-c", "--conf",
                             help="Specify configurations from a file",
                             metavar="File")
    config_file = os.path.join(os.path.dirname(__file__),
                               '../config/default_eval_config.txt')
    defaults = {"conf": config_file}
    file_parser.set_defaults(**defaults)
    file_arg, remaining_argv = file_parser.parse_known_args()

    if file_arg.conf:
        config = configparser.SafeConfigParser()
        config.read([file_arg.conf])
        defaults = dict(config.items("settings"))

    parser = argparse.ArgumentParser(
        parents=[file_parser],
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.set_defaults(**defaults)
    parser.add_argument("action",
            help="compute ROI statistics or compare segmentation maps",
            choices=['roi', 'compare'])
    parser.add_argument("--threshold",
            help="threshold to obtain binary segmentation",
            type=float)
    parser.add_argument("--step",
            help="step of increment when considering probabilistic segmentation",
            type=float)
    parser.add_argument("--ref_dir",
            help="path to the image to use as reference")
    parser.add_argument("--seg_dir",
            help="path where to find the images to evaluate")
    parser.add_argument("--img_dir",
            help="path where to find the images to evaluate")
    parser.add_argument("--save_csv_dir",
            help="path where to save the output csv file")
    parser.add_argument("--ext",
            help="extension of the image files to be read")
    parser.add_argument("--seg_type",
            help="type of input: discrete maps or probabilistic maps")
    args = parser.parse_args(remaining_argv)
    return args
