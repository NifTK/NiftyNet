# -*- coding: utf-8 -*-
"""Imports images of multiple types (2D or 3D) as `nib.Nifti1Image`"""

from collections import OrderedDict

import nibabel as nib
import numpy as np
import tensorflow as tf

from niftynet.utilities.util_import import require_module

SUPPORTED_LOADERS = OrderedDict()
AVAILABLE_LOADERS = OrderedDict()


###############################################################################
# Utility Image Loader Funtions
###############################################################################

def register_image_loader(name, requires, min_version=None, auto_discover=True):
    """
    Function decorator to register an image loader.

    SUPPORTED_LOADERS:
        Ordered dictionary were each entry is a function decorated with
        `@register_image_loader`. This is, every loader that NiftyNet supports.
        This dictionary will be dynamically filled and will be identical for
        every NiftyNet installation.

        Used only for information or error messages and logging purposes.

    AVAILABLE_LOADERS:
        A subset of the `SUPPORTED_LOADERS` that contain only the loaders that
        have the required library/module installed on the system. Dynamically
        filled from every function decorated with `@register_image_loader` that
        passes the import check. This list will be different for every
        installation, as it is platform dependant.

        Inspedted and used to load images in runtime.

    Adding a new loader only requires to decorate a function with
    `@register_image_loader` and it will populate SUPPORTED_LOADERS and
    AVAILABLE_LOADERS accordingly in runtime. The function will receive
    a filename as its only parameter, and will return an image and its
    `4x4` affinity matrix. Dummy example:

        @register_image_loader('fake', requires='numpy', min_version='1.13.3',
                               auto_discover=False)
        def imread_numpy(filename):
            np = require_module('numpy')
            return image2nibabel(np.random.rand(100, 100, 3), np.eye(4))

    It registers a loader named 'fake' that requires `numpy` version >= '1.13.3'
    to be installed. It will first dynamically load numpy library and then
    return a `(100, 100, 3)` fake color image and an identity `(4, 4)`
    affinity matrix. `loader = fake` in the data section of a config file will
    select this loader and generate fake data.

    When `auto_discover=True` (default) the method will be available to be
    automatically discovered and used if `loader` is not provided in the
    config file. This is, if no loader is specified, all the loaders
    registered with `auto_discover=True` will be looped in priority order.
    """

    def _wrapper(func):
        """Wrapper that registers a function if it satisfies requirements."""
        try:
            auto_d = auto_discover
            require_module(requires, min_version=min_version, mandatory=True)
            AVAILABLE_LOADERS[name] = dict(func=func, auto_discover=auto_d)
        except (ImportError, AssertionError):
            pass
        SUPPORTED_LOADERS[name] = (requires, min_version)
        return func

    return _wrapper


def load_image_obj(filename, loader=None):
    """
    Loads an image from a given loader or checking multiple loaders.

    If `loader` is specified the selected loader will be used if it exists in
    `AVAILABLE_LOADERS` (see above).

    If no loader is specified, all the loaders registered with
    `auto_discover=True` (default) will be looped in priority order.
    """
    if loader and loader in SUPPORTED_LOADERS:
        if loader not in AVAILABLE_LOADERS:
            raise ValueError('Image Loader {} supported but library not found.'
                             ' Required libraries: {}'
                             .format(loader, SUPPORTED_LOADERS[loader]))
        tf.logging.debug('Using requested loader: {}'.format(loader))
        loader_params = AVAILABLE_LOADERS[loader]
        return loader_params['func'](filename)
    if loader:
        raise ValueError('Image Loader {} not supported. Supported loaders: {}'
                         .format(loader, list(SUPPORTED_LOADERS.keys())))

    for name, loader_params in AVAILABLE_LOADERS.items():
        if not loader_params['auto_discover']:
            continue

        try:
            img = loader_params['func'](filename)
            tf.logging.debug('Using Image Loader {}.'.format(name))
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
        raise IOError(filename)
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
        raise IOError(filename)
    img = sitk.GetArrayFromImage(simg)
    if simg.GetDimension() > 2:
        img = img.transpose()
    return image2nibabel(img, affine=make_affine_from_sitk(simg))


@register_image_loader('dummy', requires='numpy', auto_discover=False)
def imread_numpy(filename=None):
    """Fake loader to load random data with numpy"""
    fake_img = np.random.randint(255, size=(100, 100, 3)).astype(np.uint8)
    print('test case {}', filename)
    return image2nibabel(fake_img, affine=np.eye(4))


tf.logging.info(
    'Available Image Loaders:\n{}.'.format(list(AVAILABLE_LOADERS.keys())))


###############################################################################
# Auxiliary functions
###############################################################################

def image2nibabel(img, affine=np.eye(4)):
    """
    Loads a RGB or Grayscale Image from a file and stores it in a 5D array,
    moving the color channels to the last axis for color images.
    """
    return ImageAsNibabel(img, affine)


class ImageAsNibabel(nib.Nifti1Image):
    """
    Wrapper class around a Nibabel file format. Loads an image using PIL
    (or scikit-image if available) and transforms it to a `nib.Nifti1Image`.

    The resulting 2D color image is already translated to a 5D array,
    swapping the channels to the last axis.
    """

    def __init__(self, img, affine):
        if img.ndim == 3 and img.shape[2] <= 4:  # Color Image
            img = img[:, :, None, None, :]

        if img.dtype == np.bool:  # bool is not a supported datatype by nibabel
            img = img.astype(np.uint8)

        nib.Nifti1Image.__init__(self, img, affine)


def make_affine_from_sitk(sitk_img):
    """Get affine transform in LPS"""
    if sitk_img.GetDepth() <= 0:
        return np.eye(4)

    rot = [sitk_img.TransformContinuousIndexToPhysicalPoint(p)
           for p in ((1, 0, 0),
                     (0, 1, 0),
                     (0, 0, 1),
                     (0, 0, 0))]
    rot = np.array(rot)
    affine = np.concatenate([
        np.concatenate([rot[0:3] - rot[3:], rot[3:]], axis=0),
        [[0.], [0.], [0.], [1.]]
    ], axis=1)
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine
