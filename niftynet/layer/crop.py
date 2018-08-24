# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.base_layer import Layer


class CropLayer(Layer):
    """
    This class defines a cropping operation:
    Removing ``2*border`` pixels from each spatial dim of the input,
    and return the spatially centred elements extracted from the input.
    """

    def __init__(self, border, name='crop'):
        super(CropLayer, self).__init__(name=name)
        self.border = border

    def layer_op(self, inputs):
        spatial_rank = layer_util.infer_spatial_rank(inputs)
        offsets = [0] + [int(self.border)] * spatial_rank + [0]
        # inferring the shape of the output by subtracting the border dimension
        out_shape = [
            int(d) - 2 * int(self.border)
            for d in list(inputs.shape)[1:-1]]
        out_shape = [-1] + out_shape + [-1]
        output_tensor = tf.slice(inputs, offsets, out_shape)
        return output_tensor
