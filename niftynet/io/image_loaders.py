# -*- coding: utf-8 -*-
""" Imports raw 2D images (.png; .jpg; .tiff; ...) as `nib.Nifti1Image`"""

from collections import OrderedDict

import numpy as np
import nibabel as nib

import tensorflow as tf

from niftynet.utilities.util_import import require_module


SUPPORTED_LOADERS = OrderedDict()
AVAILABLE_LOADERS = OrderedDict()

###############################################################################
# Utility Image Loader Funtions
###############################################################################

def register_image_loader(name, requires, min_version=None):
    """Function decorator to register an image loader."""
    def _wrapper(fn):
        """Wrapper that registers a function if it satisfies requirements."""
        try:
            require_module(requires, min_version=min_version)
            AVAILABLE_LOADERS[name] = fn
        except (ImportError, AssertionError):
            pass
        SUPPORTED_LOADERS[name] = (requires, min_version)
        return fn
    return _wrapper


def load_image_from_file(filename, loader=None):
    """Loads an image from a given loader or checking multiple loaders."""
    if loader is not None and loader in SUPPORTED_LOADERS:
        if loader not in AVAILABLE_LOADERS:
            raise ValueError('Image Loader {} supported buy library not found.'
                             ' Required: {}'
                              .format(loader, SUPPORTED_LOADERS[loader]))
        tf.logging.debug('Using requested loader: {}'.format(loader))
        loader = AVAILABLE_LOADERS[loader]
        return loader(filename)
    elif loader is not None:
        raise ValueError('Image Loader {} not supported. Supported loaders: {}'
                         .format(loader, list(SUPPORTED_LOADERS.keys())))

    for name, loader in AVAILABLE_LOADERS.items():
        try:
            img = loader(filename)
            tf.logging.debug('Iterating all loaders. Using: {}')
            return img
        except IOError:
            # e.g. Nibabel cannot load standard 2D images
            # e.g. PIL cannot load 16bit TIF images
            pass

    raise ValueError('No available loader could load file: {}.'
                     ' Available loaders: {}. Supported Loaders: {}'
                     .format(filename, list(AVAILABLE_LOADERS.keys()),
                             list(SUPPORTED_LOADERS.keys())))


###############################################################################
# All supported Image Loaders -- In Priority Order
###############################################################################

@register_image_loader('nibabel', requires='nibabel')
def imread_nibabel(filename):
    """Default nibabel loader for NiftyNet."""
    nib = require_module('nibabel')
    try:
        return nib.load(filename)
    except nib.filebasedimages.ImageFileError:
        raise IOError('Nibabel could not load image file: {}'.format(filename))


@register_image_loader('opencv', requires='cv2')
def imread_opencv(filename):
    """OpenCV image loader with identity 2D affine."""
    cv2 = require_module('cv2')
    img = cv2.imread(filename, flags=-1)
    if img is None:
        raise FileNotFoundError(filename)
    return image2nibabel(img[..., ::-1])


@register_image_loader('skimage', requires='skimage.io', min_version=(0, 13))
def imread_skimage(filename):
    """Scikit-image loader with an identity affine matrix."""
    skio = require_module('skimage.io')
    img = skio.imread(filename)
    return image2nibabel(img)


@register_image_loader('pillow', requires='PIL.Image')
def imread_pillow(filename):
    """PIL (Pillow) image loader with an identity affine matrix."""
    pil = require_module('PIL.Image')
    img = np.asarray(pil.open(filename))
    return image2nibabel(img)


@register_image_loader('simpleitk', requires='SimpleITK')
def imread_sitk(filename):
    """SimpleITK requires two function calls to retrieve a numpy array."""
    sitk = require_module('SimpleITK')
    try:
        simg = sitk.ReadImage(filename)
    except RuntimeError:
        raise FileNotFoundError(filename)
    img = sitk.GetArrayFromImage(simg)
    return image2nibabel(img, affine=make_affine_from_sitk(simg))


tf.logging.info('+++ Available Image Loaders {}:'
                .format(list(AVAILABLE_LOADERS.keys())))

###############################################################################
# Auxiliary functions
###############################################################################

def image2nibabel(img, affine=None):
    """
    Loads a RGB or Grayscale Image from a file and stores it in a 5D array,
    moving the color channels to the last axis for color images.
    """
    if affine is None:
        affine = make_identity_affine()
    return ImageAsNibabel(img, affine)


class ImageAsNibabel(nib.Nifti1Image):

    """
    Wrapper class around a Nibabel file format. Loads an image using PIL
    (or scikit-image if available) and transforms it to a `nib.Nifti1Image`.

    The resulting 2D image is already translated to a 5D array, swaping the
    channels to the last axis in the case of a color image.
    """
    def __init__(self, img, affine):
        if img.ndim == 3 and img.shape[2] == 3:  # Color Image
            img = img[:, :, None, None, :]
        elif img.ndim == 3: # 3D image
            img = img[:, :, :, None, None]
        elif img.ndim == 2:  # Grayscale or mask
            img = img[:, :, None, None, None]
        else:
            raise NotImplementedError

        nib.Nifti1Image.__init__(self, img, affine)


def make_identity_affine():
    """Identity affine matrix, might change in the future"""
    return np.eye(4)


def make_affine_from_sitk(simpleITKImage):
    """Get affine transform in LPS"""
    c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
         for p in ((1, 0, 0),
                   (0, 1, 0),
                   (0, 0, 1),
                   (0, 0, 0))]
    c = np.array(c)
    affine = np.concatenate([
        np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
        [[0.], [0.], [0.], [1.]]
    ], axis=1)
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine
