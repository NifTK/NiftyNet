# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import warnings

import numpy as np
import scipy.ndimage

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
        assert min_percentage < max_percentage
        self._min_percentage = max(min_percentage, -99.9)
        self._max_percentage = max_percentage
        self._rand_zoom = None

    def randomise(self, spatial_rank=3):
        spatial_rank = int(np.floor(spatial_rank))
        rand_zoom = np.random.uniform(low=self._min_percentage,
                                      high=self._max_percentage,
                                      size=(spatial_rank,))
        self._rand_zoom = (rand_zoom + 100.0) / 100.0

    def _apply_transformation(self, image, interp_order=3):
        assert self._rand_zoom is not None
        full_zoom = np.array(self._rand_zoom)
        while len(full_zoom) < image.ndim:
            full_zoom = np.hstack((full_zoom, [1.0]))

        if image.ndim == 4:
            output = []
            for mod in range(image.shape[-1]):
                scaled = scipy.ndimage.zoom(image[..., mod],
                                            full_zoom[:3],
                                            order=interp_order)
                output.append(scaled[..., np.newaxis])
            return np.concatenate(output, axis=-1)
        if image.ndim == 3:
            scaled = scipy.ndimage.zoom(image,
                                        full_zoom[:3],
                                        order=interp_order)
            return scaled[..., np.newaxis]
        raise NotImplementedError('not implemented random scaling')

    def layer_op(self, inputs, interp_orders, *args, **kwargs):
        if inputs is None:
            return inputs

        if isinstance(inputs, dict) and isinstance(interp_orders, dict):
            for (field, image) in inputs.items():
                assert image.shape[-1] == len(interp_orders[field]), \
                    "interpolation orders should be" \
                    "specified for each inputs modality"

                transformed_data = []
                for mod_i, interp_order in enumerate(interp_orders[field]):
                    scaled_data = self._apply_transformation(
                        image[..., mod_i], interp_order)
                    transformed_data.append(scaled_data[..., np.newaxis])
                inputs[field] = np.concatenate(transformed_data, axis=-1)
        else:
            raise NotImplementedError("unknown input format")
        return inputs
