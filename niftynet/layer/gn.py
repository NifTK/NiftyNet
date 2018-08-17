# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer


class GNLayer(TrainableLayer):
    """
    Group normalisation layer, with trainable mean value 'beta' and
    std 'gamma'.  'beta' is initialised to 0.0 and 'gamma' is initialised
    to 1.0.  This class assumes 'beta' and 'gamma' share the same type_str of
    regulariser.

    Reimplementation of
    Wu and He, Group Normalization, arXiv:1803.08494 (2018)
    """

    def __init__(self,
                 group_size=32,
                 regularizer=None,
                 eps=1e-5,
                 name='group_norm'):
        super(GNLayer, self).__init__(name=name)
        self.g_size = group_size
        self.eps = eps
        self.initializers = {
            'beta': tf.constant_initializer(0.0),
            'gamma': tf.constant_initializer(1.0)}
        self.regularizers = {'beta': regularizer, 'gamma': regularizer}

    def layer_op(self, inputs):
        input_shape = inputs.shape
        g_size = min(self.g_size, input_shape[-1])

        # TODO: raise an informative error when size not divisible
        grouped_shape = \
            list(input_shape[:-1]) + [g_size, input_shape[-1] // g_size]
        inputs = tf.reshape(inputs, grouped_shape)

        # operates on all dims except the grouped dim
        axes = list(range(1, input_shape.ndims - 1)) + [input_shape.ndims]

        # create the shape of trainable variables
        param_shape = [1] * (input_shape.ndims - 2) + [input_shape[-1]]

        # create trainable variables
        beta = tf.get_variable(
            'beta',
            shape=param_shape,
            initializer=self.initializers['beta'],
            regularizer=self.regularizers['beta'],
            dtype=tf.float32, trainable=True)
        gamma = tf.get_variable(
            'gamma',
            shape=param_shape,
            initializer=self.initializers['gamma'],
            regularizer=self.regularizers['gamma'],
            dtype=tf.float32, trainable=True)

        # mean and var
        mean, variance = tf.nn.moments(inputs, axes, keep_dims=True)

        outputs = (inputs - mean) / tf.sqrt(variance + self.eps)
        outputs = tf.reshape(outputs, list(input_shape)) * gamma + beta
        return outputs
