# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.base_layer import Layer


class CropLayer(Layer):
    """
    This class defines a cropping operation:
    Removing `2*border` pixels from each spatial dim of the input,
    and return the spatially centred elements extracted from the input.

    This function is implemented with a convolution in the `valid` mode
    with a trivial kernel
    """

    def __init__(self, border, name='crop'):
        super(CropLayer, self).__init__(name=name)
        self.border = border

    def layer_op(self, inputs):
        spatial_rank = layer_util.infer_spatial_rank(inputs)
        kernel_shape = np.hstack((
            [self.border * 2 + 1] * spatial_rank, 1, 1)).flatten()
        # initializer a kernel with all 0s, and set the central element to 1
        np_kernel = layer_util.trivial_kernel(kernel_shape)
        crop_kernel = tf.constant(np_kernel, dtype=inputs.dtype)
        # split channel dim
        output_tensor = [tf.expand_dims(x, -1)
                         for x in tf.unstack(inputs, axis=-1)]
        output_tensor = [tf.nn.convolution(input=inputs,
                                           filter=crop_kernel,
                                           strides=[1] * spatial_rank,
                                           padding='VALID',
                                           name='conv')
                         for inputs in output_tensor]
        output_tensor = tf.concat(output_tensor, axis=-1)
        return output_tensor
