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
    """

    def __init__(self, image_name, border, name='pad'):
        super(PadLayer, self).__init__(name=name)
        try:
            spatial_border = tuple(map(lambda x: (x,), border))
        except ValueError:
            tf.logging.fatal("unknown padding param. {}".format(border))
            raise
        self.border = spatial_border
        self.image_name = set(image_name)

    def layer_op(self, input_image, mask=None):
        if not isinstance(input_image, dict):
            full_border = match_ndim(self.border, input_image.ndim)
            input_image = np.pad(input_image, full_border, mode='minimum')
            return input_image, mask

        for name, image in input_image.items():
            if name not in self.image_name:
                tf.logging.warning(
                    'could not pad, dict name %s not in %s',
                    name, self.image_name)
                continue
            full_border = match_ndim(self.border, image.ndim)
            input_image[name] = np.pad(image, full_border, mode='minimum')
        return input_image, mask

    def inverse_op(self, input_image, mask=None):
        if not isinstance(input_image, dict):
            try:
                outputs = input_image[
                          self.border[0][0]:(-self.border[0][0]),
                          self.border[1][0]:(-self.border[1][0]),
                          self.border[2][0]:(-self.border[2][0]), ...]
            except IndexError:
                tf.logging.fatal(
                    "unable to inverse the padding "
                    "input: {}, pad param. {}".format(
                        outputs.shape, self.border))
            return outputs, mask

        for name, image in input_image.items():
            if name not in self.image_name:
                continue
            try:
                input_image[name] = image[
                                    self.border[0][0]:(-self.border[0][0]),
                                    self.border[1][0]:(-self.border[1][0]),
                                    self.border[2][0]:(-self.border[2][0]), ...]
            except IndexError:
                tf.logging.fatal(
                    "unable to inverse the padding "
                    "input: {}, pad param. {}".format(
                        image.shape, self.border))
        return input_image, mask


def match_ndim(border, image_ndim):
    full_border = border
    while len(full_border) < image_ndim:
        full_border = full_border + ((0,),)
    return full_border
