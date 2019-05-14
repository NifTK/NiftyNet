from pathlib import Path

import nibabel as nib
import numpy as np
import skimage.io as skio

from niftynet.application.application_module_wrapper import \
    SegmentationApplicationModule

IMAGES = []
LABELS = []


def get_image(idx):
    return skio.imread(IMAGES[idx])


def get_labels(idx):
    return nib.load(LABELS[idx]).get_data()


def output_image(img_out, sub, img_in):
    print("Rec'd", sub, ':', img_out)


def setup_images(args):
    global IMAGES
    global LABELS

    img_paths = [str(path) for path in Path(args.image_directory).glob('*.bmp')]
    img_paths = sorted(img_paths)

    IMAGES = []
    LABELS = []
    for img_path in img_paths:
        lab_path = Path(img_path).name.replace('.bmp', '_anno.nii.gz')
        lab_path = Path(args.labels_directory)/lab_path

        if lab_path.is_file():
            img = skio.imread(img_path)
            lab = nib.load(str(lab_path)).get_data()

            if img.shape[:2] == lab.shape[:2]:
                IMAGES.append(img_path)
                LABELS.append(str(lab_path))
            else:
                print('Inconsistent shapes: %s' % Path(img_path).name)
        else:
            print('No labels:  %s' % Path(img_path).name)


def main(args):
    setup_images(args)

    module = SegmentationApplicationModule(args.model_file)

    module.set_input_callback('slice', get_image, do_reshape_rgb=True)\
          .set_input_callback('label', get_labels, do_reshape_nd=True)\
          .set_output_callback(output_image)\
          .set_num_subjects(len(IMAGES))\
          .set_action(args.action)\
          .initialise_application()\
          .run()


if __name__ == '__main__':
    import argparse as ap

    parser = ap.ArgumentParser(description='App-as-a-Module test script')

    parser.add_argument('model_file', help='Model config.ini')
    parser.add_argument('image_directory', help='Input image directory')
    parser.add_argument('labels_directory', help='Target label directory')
    parser.add_argument('action', help='NiftyNet action')

    main(parser.parse_args())
