# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import warnings
import numpy as np

from niftynet.layer.base_layer import Layer

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


class RandomFlipLayer(Layer):
    """
     Add a random flipping layer as pre-processing.
     flip_axes: a list of indices over which to flip
     flip_probability: the probability of performing the flip
                      (default = 0.5)
    """

    def __init__(self,
                 flip_axes,
                 flip_probability=0.5,
                 name='random_flip'):
        super(RandomFlipLayer, self).__init__(name=name)
        self._flip_axes = flip_axes
        self._flip_probability = flip_probability
        self._rand_flip = None

    def randomise(self, spatial_rank=3):
        spatial_rank = int(np.floor(spatial_rank))
        _rand_flip = np.random.random(size=spatial_rank) < self._flip_probability
        self._rand_flip = _rand_flip

    def _apply_transformation(self, image):
        assert self._rand_flip is not None, "Flip is unset -- Error!"
        for axis_number, do_flip in enumerate(self._rand_flip):
            if axis_number in self._flip_axes and do_flip:
                image = np.flip(image, axis=axis_number)
        return image

    def layer_op(self, inputs):
        if inputs is None:
            return inputs
        if inputs.spatial_rank == 3:
            if inputs.data.ndim == 4:
                transformed_data = []
                for mod_i in range(inputs.data.shape[-1]):
                    scaled_data = self._apply_transformation(
                        inputs.data[..., mod_i])
                    transformed_data.append(scaled_data[..., np.newaxis])
                inputs.data = np.concatenate(transformed_data, axis=-1)
            if inputs.data.ndim == 5:
                transformed_data = []
                for t in range(inputs.data.shape[-1]):
                    mod_data = []
                    for mod_i in range(inputs.data.shape[-2]):
                        scaled_data = self._apply_transformation(
                            inputs.data[..., mod_i, t])
                        mod_data.append(scaled_data[..., np.newaxis])
                    mod_data = np.concatenate(mod_data, axis=-1)
                    transformed_data.append(mod_data[..., np.newaxis])
                inputs.data = np.concatenate(transformed_data, axis=-1)
        return inputs
