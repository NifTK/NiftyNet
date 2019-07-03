# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_OP = set(['AVG', 'MAX'])


class ChannelSELayer(Layer):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in::

         Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507
    """
    def __init__(self,
                 func='AVG',
                 reduction_ratio=16,
                 name='channel_squeeze_excitation'):
        self.func = func.upper()
        self.reduction_ratio = reduction_ratio
        super(ChannelSELayer, self).__init__(name=name)

        look_up_operations(self.func, SUPPORTED_OP)

    def layer_op(self, input_tensor):
        # spatial squeeze
        input_rank = len(input_tensor.shape)
        reduce_indices = list(range(input_rank))[1:-1]
        if self.func == 'AVG':
            squeeze_tensor = tf.reduce_mean(input_tensor, axis=reduce_indices)
        elif self.func == 'MAX':
            squeeze_tensor = tf.reduce_max(input_tensor, axis=reduce_indices)
        else:
            raise NotImplementedError("pooling function not supported")

        # channel excitation
        num_channels = int(squeeze_tensor.shape[-1])
        reduction_ratio = self.reduction_ratio
        if num_channels % reduction_ratio != 0:
            raise ValueError(
                "reduction ratio incompatible with "
                "number of input tensor channels")

        num_channels_reduced = num_channels / reduction_ratio
        fc1 = FullyConnectedLayer(num_channels_reduced,
                                  with_bias=False,
                                  feature_normalization=None,
                                  acti_func='relu',
                                  name='se_fc_1')
        fc2 = FullyConnectedLayer(num_channels,
                                  with_bias=False,
                                  feature_normalization=None,
                                  acti_func='sigmoid',
                                  name='se_fc_2')

        fc_out_1 = fc1(squeeze_tensor)
        fc_out_2 = fc2(fc_out_1)

        while len(fc_out_2.shape) < input_rank:
            fc_out_2 = tf.expand_dims(fc_out_2, axis=1)

        output_tensor = tf.multiply(input_tensor, fc_out_2)

        return output_tensor


class SpatialSELayer(Layer):
    """
    Re-implementation of SE block -- squeezing spatially
    and exciting channel-wise described in::

        Roy et al., Concurrent Spatial and Channel Squeeze & Excitation
        in Fully Convolutional Networks, arXiv:1803.02579

    """
    def __init__(self,
                 name='spatial_squeeze_excitation'):
        super(SpatialSELayer, self).__init__(name=name)

    def layer_op(self, input_tensor):
        # channel squeeze
        conv = ConvolutionalLayer(n_output_chns=1,
                                  kernel_size=1,
                                  feature_normalization=None,
                                  acti_func='sigmoid',
                                  name="se_conv")

        squeeze_tensor = conv(input_tensor)

        # spatial excitation
        output_tensor = tf.multiply(input_tensor, squeeze_tensor)

        return output_tensor


class ChannelSpatialSELayer(Layer):
    """
    Re-implementation of concurrent spatial and channel
    squeeze & excitation::

        Roy et al., Concurrent Spatial and Channel Squeeze & Excitation
        in Fully Convolutional Networks, arXiv:1803.02579

    """
    def __init__(self,
                 func='AVG',
                 reduction_ratio=16,
                 name='channel_spatial_squeeze_excitation'):
        self.func = func.upper()
        self.reduction_ratio = reduction_ratio
        super(ChannelSpatialSELayer, self).__init__(name=name)

        look_up_operations(self.func, SUPPORTED_OP)

    def layer_op(self, input_tensor):
        cSE = ChannelSELayer(func=self.func,
                             reduction_ratio=self.reduction_ratio,
                             name='cSE')
        sSE = SpatialSELayer(name='sSE')

        output_tensor = tf.add(cSE(input_tensor), sSE(input_tensor))

        return output_tensor
