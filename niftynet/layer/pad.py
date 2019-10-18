# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.base_layer import Layer, Invertible


class PadLayer(Layer, Invertible):
    """
    This class defines a padding operation:
    pad `2*border` pixels from spatial dims of the input (numpy array),
    and return the padded input.

    This function is used at volume level (as a preprocessor in image reader)
    therefore assumes the input has at least three spatial dims.
    """

    def __init__(self, image_name, border, name='pad', mode='minimum'):
        super(PadLayer, self).__init__(name=name)
        try:
            spatial_border = tuple(map(lambda x: (x,), border))
        except (ValueError, TypeError):
            tf.logging.fatal("unknown padding param. {}".format(border))
            raise
        self.border = spatial_border
        self.image_name = set(image_name)
        self.mode = mode

    def layer_op(self, input_image, mask=None):
        if not isinstance(input_image, dict):
            full_border = match_ndim(self.border, input_image.ndim)
            input_image = np.pad(input_image, full_border, mode=self.mode)
            return input_image, mask

        for name, image in input_image.items():
            if name not in self.image_name:
                tf.logging.warning(
                    'could not pad, dict name %s not in %s',
                    name, self.image_name)
                continue
            full_border = match_ndim(self.border, image.ndim)
            input_image[name] = np.pad(image, full_border, mode=self.mode)
        return input_image, mask

    def inverse_op(self, input_image, mask=None):
        if not isinstance(input_image, dict):
            full_border = match_ndim(self.border, input_image.ndim)
            outputs = _crop_numpy_array(input_image, full_border)
            return outputs, mask

        for name, image in input_image.items():
            if name not in self.image_name:
                continue
            full_border = match_ndim(self.border, image.ndim)
            input_image[name] = _crop_numpy_array(image, full_border)
        return input_image, mask


def _crop_numpy_array(image, border):
    try:
        assert image.ndim >= 3, \
            "input image must have at least 3 spatial dims"
        x_ = border[0][0] if image.shape[0] / 2 > border[0][0] > 0 else 0
        y_ = border[1][0] if image.shape[1] / 2 > border[1][0] > 0 else 0
        z_ = border[2][0] if image.shape[2] / 2 > border[2][0] > 0 else 0
        _x = -border[0][0] if image.shape[0] / 2 > border[0][0] > 0 \
            else image.shape[0]
        _y = -border[1][0] if image.shape[1] / 2 > border[1][0] > 0 \
            else image.shape[1]
        _z = -border[2][0] if image.shape[2] / 2 > border[2][0] > 0 \
            else image.shape[2]
        return image[x_:_x, y_:_y, z_:_z, ...]
    except (IndexError, AssertionError):
        tf.logging.fatal(
            "unable to inverse the padding "
            "input: {}, pad param. {}".format(
                image.shape, border))
        raise


def match_ndim(border, image_ndim):
    full_border = border
    while len(full_border) < image_ndim:
        full_border = full_border + ((0,),)
    return full_border[:image_ndim]
