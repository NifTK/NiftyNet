# -*- coding: utf-8 -*-
#Data augmentation using elastic deformations as used by:
#Milletari,F., Navab, N., & Ahmadi, S. A. (2016) V-net:
#Fully convolutional neural networks for volumetric medical
#image segmentation


from __future__ import absolute_import, print_function
from niftynet.utilities.util_import import check_module
check_module('SimpleITK')

import warnings

import numpy as np
import SimpleITK as sitk

from niftynet.layer.base_layer import RandomisedLayer

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


class RandomElasticDeformationLayer(RandomisedLayer):
    """
    generate randomised elastic deformations along each dim for data augmentation
    """

    def __init__(self,
                 num_controlpoints=4,
                 std_deformation_sigma=15,
                 name='random_elastic_deformation'):
        super(RandomElasticDeformationLayer, self).__init__(name=name)

        self.num_controlpoints = max(num_controlpoints, 2)
        self.std_deformation_sigma = max(std_deformation_sigma, 1)
        self.bspline_transformation = None

    def randomise(self, image_dict, spatial_rank=3):
        images = image_dict.values()
        equal_shapes = np.all([images[0].shape == image.shape for image in images])
        if spatial_rank == 3 and equal_shapes:
            self._randomise_bspline_transformation_3d(images[0].shape)
        else:
            # currently not supported spatial rank for elastic deformation
            print("randomising elastic deformation FAILED")
            pass

    def _randomise_bspline_transformation_3d(self, shape):
        # generate transformation
        itkimg = sitk.GetImageFromArray(np.zeros(shape[:3]))
        transfromDomainMeshSize = [self.num_controlpoints] * itkimg.GetDimension()
        self.bspline_transformation = sitk.BSplineTransformInitializer(itkimg, transfromDomainMeshSize)

        params = self.bspline_transformation.GetParameters()
        paramsNp = np.asarray(params, dtype=float)
        paramsNp = paramsNp + np.random.randn(paramsNp.shape[0]) * self.std_deformation_sigma

        # paramsNp[0:int(len(params) / 3)] = 0  # remove z deformations! The resolution in z is too bad

        params = tuple(paramsNp)
        self.bspline_transformation.SetParameters(params)

    def _apply_bspline_transformation_3d(self, image, interp_order=3):

        if (np.random.rand(1)[0] > 0.5):  # do not apply deformations always, just sometimes
            sitkImage = sitk.GetImageFromArray(image)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(sitkImage)
            if interp_order == 3:
                resampler.SetInterpolator(sitk.sitkBSpline)
            elif interp_order == 2:
                resampler.SetInterpolator(sitk.sitkLinear)
            elif interp_order == 1 or interp_order == 0:
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            else:
                raise RuntimeError("not supported interpolation_order")

            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(self.bspline_transformation)

            outimgsitk = resampler.Execute(sitkImage)

            outimg = sitk.GetArrayFromImage(outimgsitk)
            return outimg

        return image

    def layer_op(self, inputs, interp_orders, *args, **kwargs):

        if inputs is None:
            return inputs

        if isinstance(inputs, dict) and isinstance(interp_orders, dict):
            for (field, image) in inputs.items():
                assert image.shape[-1] == len(interp_orders[field]), \
                    "interpolation orders should be" \
                    "specified for each inputs modality"
                for mod_i, interp_order in enumerate(interp_orders[field]):
                    if image.ndim == 4:
                        inputs[field][..., mod_i] = \
                            self._apply_bspline_transformation_3d(
                                image[..., mod_i], interp_order)
                    elif image.ndim == 5:
                        for t in range(image.shape[-2]):
                            inputs[field][..., t, mod_i] = \
                                self._apply_bspline_transformation_3d(
                                    image[..., t, mod_i], interp_order)
                    else:
                        raise NotImplementedError("unknown input format")

        else:
            raise NotImplementedError("unknown input format")
        return inputs
