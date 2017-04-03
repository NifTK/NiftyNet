import argparse
import configparser
import os


def run():
    file_parser = argparse.ArgumentParser(add_help=False)
    file_parser.add_argument("-c", "--conf",
                             help="Specify configurations from a file",
                             metavar="File")
    config_file = os.path.join(os.path.dirname(__file__),
                               'config/default_config.txt')
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
        "--model_dir",
        metavar='',
        help="Directory to save/load intermediate training models and logs")
    parser.add_argument(
        "--net_name",
        help="choose a net from ./network/ ",
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
        "--image_size", metavar='', help="set input image size", type=int)
    parser.add_argument(
        "--label_size", metavar='', help="set label size of the net", type=int)
    parser.add_argument(
        "--num_classes", metavar='', help="set number of classes", type=int)

    parser.add_argument(
        "--volume_padding_size",
        metavar='',
        help="Set padding size of each volume (in all dimensions)",
        type=int)

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
        "--train_image_dir",
        metavar='',
        help="[Training only] Specify training input volume directory")
    parser.add_argument(
        "--train_label_dir",
        metavar='',
        help="[Training only] Training input label directory")

    parser.add_argument(
        "--lr",
        help="[Training only] set learning rate", type=float)
    parser.add_argument(
        "--decay",
        help="[Training only] set weight decay", type=float)
    parser.add_argument(
        "--loss_type",
        metavar='TYPE_STR', help="[Training only] specify loss type")
    parser.add_argument(
        "--reg_type",
        metavar='TYPE_STR', help="[Training only] specify regulariser type")
    parser.add_argument(
        "--starting_iter",
        metavar='', help="[Training only] resume from iteration n", type=int)
    parser.add_argument(
        "--save_every_n",
        metavar='', help="[Training only] Model saving frequency", type=int)
    parser.add_argument(
        "--max_iter",
        metavar='', help="[Training only] total number of iterations", type=int)

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
        "--eval_image_dir",
        metavar='',
        help="[Inference only] Directory of image to be segmented")  # without '/'
    parser.add_argument(
        "action", help="train or inference", choices=['train', 'inference'])  # without '/'
    # parser.add_argument("", help="", type=int)
    args = parser.parse_args(remaining_argv)
    return args


def run_eval():
    file_parser = argparse.ArgumentParser(add_help=False)
    file_parser.add_argument("-c", "--conf",
                             help="Specify configurations from a file",
                             metavar="File")
    config_file = os.path.join(os.path.dirname(__file__),
                               'config/default_config_eval.txt')
    defaults = {"conf": config_file}
    file_parser.set_defaults(**defaults)
    file_arg, remaining_argv = file_parser.parse_known_args()

    if file_arg.conf:
        config = ConfigParser.SafeConfigParser()
        config.read([file_arg.conf])
        defaults = dict(config.items("settings"))

    parser = argparse.ArgumentParser(
        parents=[file_parser],
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.set_defaults(**defaults)
    parser.add_argument("--threshold",help="threshold to obtain binary segmentation")
    parser.add_argument("--step",help="step of increment when considering probabilistic segmentation")
    parser.add_argument("--ref_image_dir",help = "path to the image to use as reference")
    parser.add_argument("--seg_image_dir", help="path where to find the images to evaluate")
    parser.add_argument("--save_eval_dir", help="path where to save the output csv file")
    parser.add_argument("--name_out", help="string to append for the naming of the output file")
    parser.add_argument("--ext", help="extension of the image files to be read")
    parser.add_argument("--list_file", help="Text file containing the list of names to use")
    parser.add_argument("--name_ref", help="ID for a specific reference file to use")
    args = parser.parse_args(remaining_argv)
    return args

def run_stats():
    file_parser = argparse.ArgumentParser(add_help=False)
    file_parser.add_argument("-c", "--conf",
                             help="Specify configurations from a file",
                             metavar="File")
    config_file = os.path.join(os.path.dirname(__file__),
                               'config/default_config_stats.txt')
    defaults = {"conf": config_file}
    file_parser.set_defaults(**defaults)
    file_arg, remaining_argv = file_parser.parse_known_args()

    if file_arg.conf:
        config = ConfigParser.SafeConfigParser()
        config.read([file_arg.conf])
        defaults = dict(config.items("settings"))

    parser = argparse.ArgumentParser(
        parents=[file_parser],
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.set_defaults(**defaults)
    parser.add_argument("--threshold",help="threshold to obtain binary segmentation")
    parser.add_argument("--step",help="step of increment when considering probabilistic segmentation")
    parser.add_argument("--data_image_dir",help = "path to the image to use for stats")
    parser.add_argument("--seg_image_dir", help="path where to find the images to evaluate")
    parser.add_argument("--save_out_dir", help="path where to save the output csv file")
    parser.add_argument("--name_out", help="string to append for the naming of the output file")
    parser.add_argument("--ext", help="extension of the image files to be read")
    parser.add_argument("--list_file", help="Text file containing the list of names to use")
    parser.add_argument("--name_seg", help="ID for a specific segmentation file to use")
    parser.add_argument("--type_stats", help="Type of analysis to be performed")
    args = parser.parse_args(remaining_argv)
    return args
