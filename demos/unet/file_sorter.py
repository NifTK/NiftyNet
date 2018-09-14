import os
from shutil import copy
import argparse


def get_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir',
                        default=os.path.join('data', 'u-net'),
                        help='The directory containing the cell tracking data.',
                        )
    parser.add_argument('--experiment_names',
                        default=['DIC-C2DH-HeLa', 'PhC-C2DH-U373'],
                        help='The names of the cell tracking experiments.',
                        type=list
                        )
    return parser.parse_args()


def main():
    args = get_user_args()

    # for each specified folder
    for experiment_name in args.experiment_names:
        out_dir = os.path.join(args.file_dir, experiment_name, 'niftynet_data')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        for root, _, files in os.walk(os.path.join(args.file_dir, experiment_name)):
            for name in [f for f in files if 'track' not in f]:
                if 'niftynet_data' not in root:  # don't look at the ones that are done already
                    cell_id = root.split(os.sep)[root.split('/').index(experiment_name) + 1][:2]
                    out_name = name.replace('t0', 'img_0').replace('t1', 'img_1').replace('man_seg', 'seg_')
                    out_name = ''.join([out_name.split('.')[0] + '_', cell_id, '.tif'])
                    out_path = os.path.join(out_dir, out_name)
                    copy(os.path.join(root, name), out_path)


if __name__ == "__main__":
    main()
