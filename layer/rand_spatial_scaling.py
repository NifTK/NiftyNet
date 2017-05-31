# -*- coding: utf-8 -*-
import warnings
import tensorflow as tf
import numpy as np
import scipy.ndimage

from .base_layer import Layer

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


class RandomSpatialScalingLayer(Layer):
    """
    generate randomised scaling along each dim for data augmentation
    """

    def __init__(self,
                 max_percentage=10,
                 name='random_spatial_scaling'):
        super(RandomSpatialScalingLayer, self).__init__(name=name)
        if max_percentage < 0:
            self._max_percentage = -1 * int(max_percentage)
        else:
            self._max_percentage = int(max_percentage)
        self._rand_zoom = None

    def randomise(self, spatial_rank=3):
        rand_zoom = np.random.uniform(low=-self._max_percentage,
                                      high=self._max_percentage,
                                      size=(spatial_rank,))
        self._rand_zoom = (rand_zoom + 100.0) / 100.0

    def _apply_transformation(self, image, interp_order=3):
        assert self._rand_zoom is not None
        full_zoom = np.array(self._rand_zoom)
        while len(full_zoom) < image.ndim:
            full_zoom = np.hstack((full_zoom, [1.0]))
        image = scipy.ndimage.zoom(image, full_zoom, order=interp_order)
        return image

    def layer_op(self, inputs):
        if inputs is None:
            return inputs
        if inputs.spatial_rank == 3:
            if inputs.data.ndim == 4:
                transformed_data = []
                for mod_i in range(inputs.data.shape[-1]):
                    scaled_data = self._apply_transformation(
                        inputs.data[..., mod_i], inputs.interp_order)
                    transformed_data.append(scaled_data[..., np.newaxis])
                inputs.data = np.concatenate(transformed_data, axis=-1)
            if inputs.data.ndim == 5:
                transformed_data = []
                for t in range(inputs.data.shape[-1]):
                    mod_data = []
                    for mod_i in range(inputs.data.shape[-2]):
                        scaled_data = self._apply_transformation(
                            inputs.data[..., mod_i, t], inputs.interp_order)
                        mod_data.append(scaled_data[..., np.newaxis])
                    mod_data = np.concatenate(mod_data, axis=-1)
                    transformed_data.append(mod_data[..., np.newaxis])
                inputs.data = np.concatenate(transformed_data, axis=-1)
        if inputs.interp_order > 0:
            inputs.data = inputs.data.astype(np.float)
        elif inputs.interp_order == 0:
            inputs.data = inputs.data.astype(np.int64)
        else:
            raise ValueError('negative interpolation order')
        return inputs
