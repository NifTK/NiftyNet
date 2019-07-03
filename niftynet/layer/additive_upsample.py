# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.linear_resize import LinearResizeLayer as ResizingLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer as Deconv
from niftynet.layer.layer_util import check_divisible_channels
from niftynet.layer.elementwise import ElementwiseLayer


class AdditiveUpsampleLayer(Layer):
    """
    Implementation of bilinear (or trilinear) additive upsampling layer,
    described in paper:

        Wojna et al., The devil is in the decoder,
        https://arxiv.org/abs/1707.05847

    In the paper 2D images are upsampled by a factor of 2 and ``n_splits = 4``
    """

    def __init__(self, new_size, n_splits, name='linear_additive_upsample'):
        """

        :param new_size: integer or a list of integers set the output
            2D/3D spatial shape.  If the parameter is an integer ``d``,
            it'll be expanded to ``(d, d)`` and ``(d, d, d)`` for 2D and
            3D inputs respectively.
        :param n_splits: integer, the output tensor will have ``C / n_splits``
            channels, where ``C`` is the number of channels of the input.
            (``n_splits`` must evenly divide ``C``.)
        :param name: (optional) name of the layer
        """
        super(AdditiveUpsampleLayer, self).__init__(name=name)
        self.new_size = new_size
        self.n_splits = int(n_splits)

    def layer_op(self, input_tensor):
        """
        If the input has the shape ``batch, X, Y,[ Z,] Channels``,
        the output will be
        ``batch, new_size_x, new_size_y,[ new_size_z,] channels/n_splits``.

        :param input_tensor: 2D/3D image tensor, with shape:
            ``batch, X, Y,[ Z,] Channels``
        :return: linearly additively upsampled volumes
        """
        check_divisible_channels(input_tensor, self.n_splits)

        resizing_layer = ResizingLayer(self.new_size)
        split = tf.split(resizing_layer(input_tensor), self.n_splits, axis=-1)
        split_tensor = tf.stack(split, axis=-1)
        output_tensor = tf.reduce_sum(split_tensor, axis=-1)
        return output_tensor


class ResidualUpsampleLayer(TrainableLayer):
    """
    Implementation of the upsampling layer with residual like connections,
    described in paper:

        Wojna et al., The devil is in the decoder,
        https://arxiv.org/abs/1707.05847

    """

    def __init__(self,
                 kernel_size=3,
                 stride=2,
                 n_splits=2,
                 w_initializer=None,
                 w_regularizer=None,
                 acti_func='relu',
                 name='residual_additive_upsample'):
        TrainableLayer.__init__(self, name=name)
        self.n_splits = n_splits
        self.deconv_param = {'w_initializer': w_initializer,
                             'w_regularizer': w_regularizer,
                             'kernel_size': kernel_size,
                             'acti_func': acti_func,
                             'stride': stride}

    def layer_op(self, input_tensor, is_training=True):
        """
        output is an elementwise sum of deconvolution and additive upsampling::

            --(inputs)--o--deconvolution-------+--(outputs)--
                        |                      |
                        o--additive upsampling-o
        :param input_tensor:
        :param is_training:
        :return: an upsampled tensor with ``n_input_channels/n_splits``
            feature channels.
        """
        n_output_chns = check_divisible_channels(input_tensor, self.n_splits)
        # deconvolution path
        deconv_output = Deconv(n_output_chns=n_output_chns,
                               with_bias=False, feature_normalization='batch',
                               **self.deconv_param)(input_tensor, is_training)

        # additive upsampling path
        additive_output = AdditiveUpsampleLayer(
            new_size=deconv_output.get_shape().as_list()[1:-1],
            n_splits=self.n_splits)(input_tensor)

        output_tensor = ElementwiseLayer('SUM')(deconv_output, additive_output)
        return output_tensor
