# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.deconvolution import DeconvLayer
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_OP = set(['REPLICATE', 'CHANNELWISE_DECONV'])


class UpSampleLayer(TrainableLayer):
    """
    This class defines channel-wise upsampling operations.
    Different from ``DeconvLayer``,
    the elements are not mixed in the channel dim.

    ``REPLICATE`` mode replicates each spatial_dim into
    ``spatial_dim*kernel_size``
    `CHANNELWISE_DECONV`` mode makes a projection using a kernel.
    e.g., With 2D input (without loss of generality), given input
    ``[N, X, Y, C]``, the output is ``[N, X*kernel_size, Y*kernel_size, C]``.
    """

    def __init__(self,
                 func,
                 kernel_size=3,
                 stride=2,
                 w_initializer=None,
                 w_regularizer=None,
                 with_bias=False,
                 b_initializer=None,
                 b_regularizer=None,
                 name='upsample'):
        self.func = look_up_operations(func.upper(), SUPPORTED_OP)
        self.layer_name = '{}_{}'.format(self.func.lower(), name)
        super(UpSampleLayer, self).__init__(name=self.layer_name)

        self.kernel_size = kernel_size
        self.stride = stride
        self.with_bias = with_bias

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor):
        spatial_rank = layer_util.infer_spatial_rank(input_tensor)
        output_tensor = input_tensor
        if self.func == 'REPLICATE':
            if self.kernel_size != self.stride:
                raise ValueError(
                    "`kernel_size` != `stride` currently not"
                    "supported in `REPLICATE` mode. Please"
                    "consider using `CHANNELWISE_DECONV` operation.")
            # simply replicate input values to
            # local regions of (kernel_size ** spatial_rank) element
            kernel_size_all_dims = layer_util.expand_spatial_params(
                self.kernel_size, spatial_rank)
            pixel_num = np.prod(kernel_size_all_dims)
            repmat = np.hstack((pixel_num, [1] * spatial_rank, 1)).flatten()
            output_tensor = tf.tile(input=input_tensor, multiples=repmat)
            output_tensor = tf.batch_to_space_nd(
                input=output_tensor,
                block_shape=kernel_size_all_dims,
                crops=[[0, 0]] * spatial_rank)

        elif self.func == 'CHANNELWISE_DECONV':
            output_tensor = [tf.expand_dims(x, -1)
                             for x in tf.unstack(input_tensor, axis=-1)]
            output_tensor = [DeconvLayer(n_output_chns=1,
                                         kernel_size=self.kernel_size,
                                         stride=self.stride,
                                         padding='SAME',
                                         with_bias=self.with_bias,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         b_initializer=self.initializers['b'],
                                         b_regularizer=self.regularizers['b'],
                                         name='deconv_{}'.format(i))(x)
                             for (i, x) in enumerate(output_tensor)]
            output_tensor = tf.concat(output_tensor, axis=-1)
        return output_tensor
