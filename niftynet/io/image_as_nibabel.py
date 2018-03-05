# -*- coding: utf-8 -*-
""" Imports raw 2D images (.png; .jpg; .tiff; ...) as `nib.Nifti1Image`"""

import numpy as np
import nibabel as nib

import tensorflow as tf

from PIL import Image

try:
    # Check whether scikit-image is installed in the system and use it.
    # It loads images slightly faster then PIL.
    # Only use scikit-image if installed version is 0.13.0 or newer
    from niftynet.utilities.util_import import require_module
    SKIO = require_module('skimage.io', (0, 13, 0))
    USE_SKIMAGE = True
    tf.logging.info('+++ Using SKIMAGE as Image Loading backend')
except (ImportError, AssertionError):
    tf.logging.info('+++ Using PIL as Image Loading backend')
    USE_SKIMAGE = False


def image2nibabel(filename):
    """
    Loads a RGB or Grayscale Image from a file and stores it in a 5D array,
    moving the color channels to the last axis for color images.
    """
    return ImageAsNibabel(filename)


class ImageAsNibabel(nib.Nifti1Image):

    """
    Wrapper class around a Nibabel file format. Loads an image using PIL
    (or scikit-image if available) and transforms it to a `nib.Nifti1Image`.

    The resulting 2D image is already translated to a 5D array, swaping the
    channels to the last axis in the case of a color image.
    """
    def __init__(self, filename):
        if USE_SKIMAGE:
            img = SKIO.imread(filename)
        else:
            img = Image.open(filename)
            img = np.asarray(img)

        if img.ndim == 3:  # Color Image: move color channels to last dimensions
            img = img[:, :, None, None, :]
        else:  # Grayscale or mask, make it 5D
            img = img[:, :, None, None, None]

        affine = np.eye(4)
        nib.Nifti1Image.__init__(self, img, affine)
