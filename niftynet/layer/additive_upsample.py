# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.layer.linear_resize import LinearResizeLayer as ResizingLayer


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
        n_channels = input_tensor.shape.as_list()[-1]
        assert self.n_splits > 0 and n_channels % self.n_splits == 0, \
            "Number of feature channels should be divisible by n_splits"

        resizing_layer = ResizingLayer(self.new_size)
        split = tf.split(resizing_layer(input_tensor), self.n_splits, axis=-1)
        split_tensor = tf.stack(split, axis=-1)
        output_tensor = tf.reduce_sum(split_tensor, axis=-1)
        return output_tensor
