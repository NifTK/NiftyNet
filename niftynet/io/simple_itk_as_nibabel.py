# -*- coding: utf-8 -*-
"""
Loading SimpleITK image as a Nibabel object.
"""
from niftynet.utilities.util_import import check_module

check_module('SimpleITK')

import SimpleITK as sitk
import nibabel
import numpy as np


class SimpleITKAsNibabel(nibabel.Nifti1Image):
    """
    Minimal interface to use a SimpleITK image as if it were
    a nibabel object. Currently only supports the subset of the
    interface used by NiftyNet and is read only
    """

    def __init__(self, filename):
        try:
            self._SimpleITKImage = sitk.ReadImage(filename)
        except RuntimeError as err:
            if 'Unable to determine ImageIO reader' in str(err):
                raise nibabel.filebasedimages.ImageFileError(str(err))
            else:
                raise
        # self._header = SimpleITKAsNibabelHeader(self._SimpleITKImage)
        affine = make_affine(self._SimpleITKImage)
        # super(SimpleITKAsNibabel, self).__init__(
        #     sitk.GetArrayFromImage(self._SimpleITKImage).transpose(), affine)
        nibabel.Nifti1Image.__init__(
            self,
            sitk.GetArrayFromImage(self._SimpleITKImage).transpose(), affine)


class SimpleITKAsNibabelHeader(nibabel.spatialimages.SpatialHeader):
    def __init__(self, image_reference):
        super(SimpleITKAsNibabelHeader, self).__init__(
            data_dtype=sitk.GetArrayViewFromImage(image_reference).dtype,
            shape=sitk.GetArrayViewFromImage(image_reference).shape,
            zooms=image_reference.GetSpacing())


def make_affine(simpleITKImage):
    # get affine transform in LPS
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
