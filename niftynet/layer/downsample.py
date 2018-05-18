# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.base_layer import Layer
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_OP = set(['AVG', 'MAX', 'CONSTANT'])
SUPPORTED_PADDING = set(['SAME', 'VALID'])


class DownSampleLayer(Layer):
    def __init__(self,
                 func,
                 kernel_size=3,
                 stride=2,
                 padding='SAME',
                 name='pooling'):
        self.func = func.upper()
        self.layer_name = '{}_{}'.format(self.func.lower(), name)
        super(DownSampleLayer, self).__init__(name=self.layer_name)

        self.padding = padding.upper()
        look_up_operations(self.padding, SUPPORTED_PADDING)

        self.kernel_size = kernel_size
        self.stride = stride

    def layer_op(self, input_tensor):
        spatial_rank = layer_util.infer_spatial_rank(input_tensor)
        look_up_operations(self.func, SUPPORTED_OP)
        kernel_size_all_dims = layer_util.expand_spatial_params(
            self.kernel_size, spatial_rank)
        stride_all_dims = layer_util.expand_spatial_params(
            self.stride, spatial_rank)
        if self.func == 'CONSTANT':
            full_kernel_size = kernel_size_all_dims + (1, 1)
            np_kernel = layer_util.trivial_kernel(full_kernel_size)
            kernel = tf.constant(np_kernel, dtype=tf.float32)
            output_tensor = [tf.expand_dims(x, -1)
                             for x in tf.unstack(input_tensor, axis=-1)]
            output_tensor = [
                tf.nn.convolution(
                    input=inputs,
                    filter=kernel,
                    strides=stride_all_dims,
                    padding=self.padding,
                    name='conv')
                for inputs in output_tensor]
            output_tensor = tf.concat(output_tensor, axis=-1)
        else:
            output_tensor = tf.nn.pool(
                input=input_tensor,
                window_shape=kernel_size_all_dims,
                pooling_type=self.func,
                padding=self.padding,
                dilation_rate=[1] * spatial_rank,
                strides=stride_all_dims,
                name=self.layer_name)
        return output_tensor
