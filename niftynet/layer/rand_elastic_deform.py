# -*- coding: utf-8 -*-
"""
Data augmentation using elastic deformations as used by:
Milletari,F., Navab, N., & Ahmadi, S. A. (2016) V-net:
Fully convolutional neural networks for volumetric medical
image segmentation
"""

from __future__ import absolute_import, print_function

import warnings

import numpy as np

from niftynet.layer.base_layer import RandomisedLayer
from niftynet.utilities.util_import import require_module

sitk = require_module('SimpleITK')

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


class RandomElasticDeformationLayer(RandomisedLayer):
    """
    generate randomised elastic deformations
    along each dim for data augmentation
    """

    def __init__(self,
                 num_controlpoints=4,
                 std_deformation_sigma=15,
                 proportion_to_augment=0.5,
                 spatial_rank=3):
        """
        This layer elastically deforms the inputs,
        for data-augmentation purposes.

        :param num_controlpoints:
        :param std_deformation_sigma:
        :param proportion_to_augment: what fraction of the images
            to do augmentation on
        :param name: name for tensorflow graph
        (may be computationally expensive).
        """

        super(RandomElasticDeformationLayer, self).__init__(
            name='random_elastic_deformation')

        self._bspline_transformation = None
        self.num_controlpoints = max(num_controlpoints, 2)
        self.std_deformation_sigma = max(std_deformation_sigma, 1)
        self.proportion_to_augment = proportion_to_augment
        if not sitk:
            self.proportion_to_augment = -1
        self.spatial_rank = spatial_rank

    def randomise(self, image_dict):
        images = list(image_dict.values())
        equal_shapes = np.all(
            [images[0].shape[:self.spatial_rank] == image.shape[:self.spatial_rank] for image in images])
        if equal_shapes and self.proportion_to_augment >= 0:
            self._randomise_bspline_transformation(images[0].shape)
        else:
            # currently not supported spatial rank for elastic deformation
            # should support classification in the future
            print("randomising elastic deformation FAILED")
            pass

    def _randomise_bspline_transformation(self, shape):
        # generate transformation
        if len(shape) == 5:  # for niftynet reader outputs
            squeezed_shape = [dim for dim in shape[:3] if dim > 1]
        else:
            squeezed_shape = shape[:self.spatial_rank]
        itkimg = sitk.GetImageFromArray(np.zeros(squeezed_shape))
        trans_from_domain_mesh_size = \
            [self.num_controlpoints] * itkimg.GetDimension()
        self._bspline_transformation = sitk.BSplineTransformInitializer(
            itkimg, trans_from_domain_mesh_size)

        params = self._bspline_transformation.GetParameters()
        params_numpy = np.asarray(params, dtype=float)
        params_numpy = params_numpy + np.random.randn(
            params_numpy.shape[0]) * self.std_deformation_sigma

        # remove z deformations! The resolution in z is too bad
        # params_numpy[0:int(len(params) / 3)] = 0

        params = tuple(params_numpy)
        self._bspline_transformation.SetParameters(params)

    def _apply_bspline_transformation(self, image, interp_order=3):
        """
        Apply randomised transformation to 2D or 3D image

        :param image: 2D or 3D array
        :param interp_order: order of interpolation
        :return: the transformed image
        """
        resampler = sitk.ResampleImageFilter()
        if interp_order > 1:
            resampler.SetInterpolator(sitk.sitkBSpline)
        elif interp_order == 1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif interp_order == 0:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            return image

        squeezed_image = np.squeeze(image)
        while squeezed_image.ndim < self.spatial_rank:
            # pad to the required number of dimensions
            squeezed_image = squeezed_image[..., None]
        sitk_image = sitk.GetImageFromArray(squeezed_image)
        
        resampler.SetReferenceImage(sitk_image)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(self._bspline_transformation)
        out_img_sitk = resampler.Execute(sitk_image)
        out_img = sitk.GetArrayFromImage(out_img_sitk)
        return out_img.reshape(image.shape)

    def layer_op(self, inputs, interp_orders, *args, **kwargs):
        if inputs is None:
            return inputs

        # only do augmentation with a probability `proportion_to_augment`
        do_augmentation = np.random.rand() < self.proportion_to_augment
        if not do_augmentation:
            return inputs

        if isinstance(inputs, dict) and isinstance(interp_orders, dict):
            for (field, image) in inputs.items():
                assert image.shape[-1] == len(interp_orders[field]), \
                    "interpolation orders should be" \
                    "specified for each inputs modality"
                for mod_i, interp_order in enumerate(interp_orders[field]):
                    if image.ndim in (3, 4):  # for 2/3d images
                        inputs[field][..., mod_i] = \
                            self._apply_bspline_transformation(
                                image[..., mod_i], interp_order)
                    elif image.ndim == 5:
                        for t in range(image.shape[-2]):
                            inputs[field][..., t, mod_i] = \
                                self._apply_bspline_transformation(
                                    image[..., t, mod_i], interp_order)
                    else:
                        raise NotImplementedError("unknown input format")

        else:
            raise NotImplementedError("unknown input format")
        return inputs
