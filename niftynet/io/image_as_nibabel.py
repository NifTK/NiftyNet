# -*- coding: utf-8 -*-
""" Imports raw 2D images (.png; .jpg; .tiff; ...) as `nib.Nifti1Image`"""

from collections import OrderedDict

import numpy as np
import nibabel as nib

import tensorflow as tf

from niftynet.utilities.util_import import require_module


def imread_sitk(filename):
    """SimpleITK requires two function calls to retrieve a numpy array."""
    sitk = require_module('SimpleITK')
    return sitk.GetArrayFromImage(sitk.ReadImage(filename))


EXTERNAL_LOADERS = [
    dict(name='opencv', module='cv2', method='imread', kwargs=dict(flags=-1)),
    dict(name='skimage', module='skimage.io', method='imread', version='0.13'),
    dict(name='pillow', module='PIL.Image', method='open'),
    dict(name='sitk', module='SimpleITK', method='ReadImage', fn=imread_sitk),
]

AVAILABLE_LOADER = OrderedDict()

for _loader in EXTERNAL_LOADERS:
    try:
        # Default params
        min_ver = _loader.get('version', None)
        args = _loader.get('args', tuple())
        kwargs = _loader.get('kwargs', dict())
        # Check the external module exists and contains the required method
        external_module = require_module(_loader['module'], min_version=min_ver)
        # retrieve external method
        external_method = getattr(external_module, _loader['method'])
        loader_function = _loader.get('fn', external_method)
        # Save loader params
        loader_dict = dict(fn=loader_function, args=args, kwargs=kwargs)
    except (ImportError, AssertionError, AttributeError):
        continue

    AVAILABLE_LOADER[_loader['name']] = loader_dict


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
