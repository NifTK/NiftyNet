# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import warnings

import numpy as np

from niftynet.layer.base_layer import RandomisedLayer

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)


class RandomFlipLayer(RandomisedLayer):
    """
    Add a random flipping layer as pre-processing.
    """

    def __init__(self,
                 flip_axes,
                 flip_probability=0.5,
                 name='random_flip'):
        """

        :param flip_axes: a list of indices over which to flip
        :param flip_probability: the probability of performing the flip
            (default = 0.5)
        :param name:
        """
        super(RandomFlipLayer, self).__init__(name=name)
        self._flip_axes = flip_axes
        self._flip_probability = flip_probability
        self._rand_flip = None

    def randomise(self, spatial_rank=3):
        spatial_rank = int(np.floor(spatial_rank))
        self._rand_flip = np.random.random(
            size=spatial_rank) < self._flip_probability

    def _apply_transformation(self, image):
        assert self._rand_flip is not None, "Flip is unset -- Error!"
        for axis_number, do_flip in enumerate(self._rand_flip):
            if axis_number in self._flip_axes and do_flip:
                image = np.flip(image, axis=axis_number)
        return image

    def layer_op(self, inputs, *args, **kwargs):
        if inputs is None:
            return inputs
        if isinstance(inputs, dict):
            for (field, image_data) in inputs.items():
                inputs[field] = self._apply_transformation(image_data)
        else:
            inputs = self._apply_transformation(inputs)
        return inputs
