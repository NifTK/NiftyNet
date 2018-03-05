# -*- coding: utf-8 -*-
""" Imports raw 2D images (.png; .jpg; .tiff; ...) as `nib.Nifti1Image`"""

import numpy as np
import nibabel as nib

import tensorflow as tf

from collections import OrderedDict

from niftynet.utilities.util_import import require_module


EXTERNAL_LOADERS = [
    dict(name='opencv', module='cv2', method='imread', kwargs=dict(flags=-1)),
    dict(name='skimage', module='skimage.io', method='imread', version='0.13'),
    dict(name='pillow', module='PIL.Image', method='open')
]

AVAILABLE_LOADER = OrderedDict()

for loader in EXTERNAL_LOADERS:
    try:
        # Default params
        min_ver = loader.get('version', None)
        args = loader.get('args', tuple())
        kwargs = loader.get('kwargs', dict())
        # Retrieve external function
        external_module = require_module(loader['module'], min_version=min_ver)
        external_function = getattr(external_module, loader['method'])
        # Save loader params
        loader_dict = dict(fn=external_function, args=args, kwargs=kwargs)
    except (ImportError, AssertionError, AttributeError):
        continue

    AVAILABLE_LOADER[loader['name']] = loader_dict


tf.logging.info('+++ Available Image Loaders {}:'
                 .format(list(AVAILABLE_LOADER.keys())))


def image2nibabel(filename, loader=None):
    """
    Loads a RGB or Grayscale Image from a file and stores it in a 5D array,
    moving the color channels to the last axis for color images.
    """
    return ImageAsNibabel(filename, loader=loader)


class ImageAsNibabel(nib.Nifti1Image):

    """
    Wrapper class around a Nibabel file format. Loads an image using PIL
    (or scikit-image if available) and transforms it to a `nib.Nifti1Image`.

    The resulting 2D image is already translated to a 5D array, swaping the
    channels to the last axis in the case of a color image.
    """
    def __init__(self, filename, loader=None, affine=np.eye(4)):
        if len(AVAILABLE_LOADER) == 0:
            ext_loaders = [v['name'] for v in EXTERNAL_LOADERS]
            raise ImportError('No supported library for loading 2D images was'
                              ' found installed in your system. Supported 2D'
                              ' image loaders: {}'.format(ext_loaders))
        elif loader is None:
            loader = AVAILABLE_LOADER[next(iter(AVAILABLE_LOADER))]
        elif loader in AVAILABLE_LOADER:
            loader = AVAILABLE_LOADER[loader]
        else:
            raise ValueError('Image loader {} not supported. Supported Image'
                             ' loaders are as follows: {}'
                             .format(loader, list(AVAILABLE_LOADER.keys())))

        img = loader['fn'](filename, *loader['args'], **loader['kwargs'])
        img = np.asarray(img)

        if img.ndim == 3:  # Color Image: move color channels to last dimensions
            img = img[:, :, None, None, :]
        else:  # Grayscale or mask, make it 5D
            img = img[:, :, None, None, None]

        nib.Nifti1Image.__init__(self, img, affine)
