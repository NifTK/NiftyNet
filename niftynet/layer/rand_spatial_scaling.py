# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import warnings

import numpy as np
import scipy.ndimage as ndi

from niftynet.layer.base_layer import RandomisedLayer

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


class RandomSpatialScalingLayer(RandomisedLayer):
    """
    generate randomised scaling along each dim for data augmentation
    """

    def __init__(self,
                 min_percentage=-10.0,
                 max_percentage=10.0,
                 name='random_spatial_scaling'):
        super(RandomSpatialScalingLayer, self).__init__(name=name)
        assert min_percentage <= max_percentage
        self._min_percentage = max(min_percentage, -99.9)
        self._max_percentage = max_percentage
        self._rand_zoom = None

    def randomise(self, spatial_rank=3):
        spatial_rank = int(np.floor(spatial_rank))
        rand_zoom = np.random.uniform(low=self._min_percentage,
                                      high=self._max_percentage,
                                      size=(spatial_rank,))
        self._rand_zoom = (rand_zoom + 100.0) / 100.0

    def _get_sigma(self, zoom):
        """
        Compute optimal standard deviation for Gaussian kernel.

            Cardoso et al., "Scale factor point spread function matching:
            beyond aliasing in image resampling", MICCAI 2015
        """
        k = 1 / zoom
        variance = (k ** 2 - 1 ** 2) * (2 * np.sqrt(2 * np.log(2))) ** (-2)
        sigma = np.sqrt(variance)
        return sigma

    def _apply_transformation(self, image, interp_order=3):
        if interp_order < 0:
            return image
        assert self._rand_zoom is not None
        full_zoom = np.array(self._rand_zoom)
        while len(full_zoom) < image.ndim:
            full_zoom = np.hstack((full_zoom, [1.0]))
        smooth = all(full_zoom[:3] < 1)  # perform smoothing if undersampling
        smooth = False
        if smooth:
            sigma = self._get_sigma(full_zoom[:3])
        if image.ndim == 4:
            output = []
            for mod in range(image.shape[-1]):
                if smooth:
                    original_resolution = ndi.gaussian_filter(
                        image[..., mod], sigma)
                else:
                    original_resolution = image[..., mod]
                scaled = ndi.zoom(
                    original_resolution, full_zoom[:3], order=interp_order)
                output.append(scaled[..., np.newaxis])
            return np.concatenate(output, axis=-1)
        elif image.ndim == 3:
            if smooth:
                original_resolution = ndi.gaussian_filter(
                    image, sigma)
            else:
                original_resolution = image
            scaled = ndi.zoom(
                original_resolution, full_zoom[:3], order=interp_order)
            return scaled[..., np.newaxis]
        else:
            raise NotImplementedError('not implemented random scaling')

    def layer_op(self, inputs, interp_orders, *args, **kwargs):
        if inputs is None:
            return inputs

        if isinstance(inputs, dict) and isinstance(interp_orders, dict):

            for (field, image) in inputs.items():
                transformed_data = []
                interp_order = interp_orders[field][0]
                for mod_i in range(image.shape[-1]):
                    scaled_data = self._apply_transformation(
                        image[..., mod_i], interp_order)
                    transformed_data.append(scaled_data[..., np.newaxis])
                inputs[field] = np.concatenate(transformed_data, axis=-1)
            # shapes = []
            # for (field, image) in inputs.items():
            #     shapes.append(image.shape)
            # assert(len(shapes) == 2 and shapes[0][0:4] == shapes[1][0:4]), shapes
        else:
            raise NotImplementedError("unknown input format")
        return inputs
