

import numpy as np
import nibabel as nib

import tensorflow as tf

from PIL import Image

try:
    # Check whether scikit-image is installed in the system and use it.
    # It loads images slightly faster then PIL.
    # Only use scikit-image if installed version is 0.13.0 or newer
    from skimage import __version__, io as skio
    USE_SKIMAGE = tuple(map(int, __version__.split('.'))) >= (0, 13, 0)
    tf.logging.info('+++ Using SKIMAGE as Image Loading backend')
except ImportError:
    tf.logging.info('+++ Using PIL as Image Loading backend')
    USE_SKIMAGE = False


def image2nibabel(filename):
    """
    Loads a RGB or Grayscale Image from a file and stores it in a 5D vector,
    moving the color channels to the last axis for color images.
    """
    return ImageAsNibabel(filename)


class ImageAsNibabel(nib.Nifti1Image):

    def __init__(self, filename):
        if USE_SKIMAGE:
            img = skio.imread(filename)
        else:
            img = Image.open(filename)
            img = np.asarray(img)

        if img.ndim == 3:  # Color Image: move color channels to last dimensions
            img = img[:, :, None, None, :]
        else:  # Grayscale or mask, make it 5D
            img = img[:, :, None, None, None]

        affine = np.eye(4)
        nib.Nifti1Image.__init__(self, img, affine)
