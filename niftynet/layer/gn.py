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
    """

    def __init__(self,
                 group=32,
                 regularizer=None,
                 eps=1e-5,
                 name='group_norm'):
        super(GNLayer, self).__init__(name=name)
        
        self.group = group
        
        self.eps = eps

        self.initializers = {'beta': tf.constant_initializer(0.0),
                             'gamma': tf.constant_initializer(1.0)}

        self.regularizers = {'beta': regularizer, 'gamma': regularizer}

    def layer_op(self, inputs, use_local_stats=False):
        input_shape = inputs.shape
        
        group = min(self.group, input_shape[-1])
        
        inputs = tf.reshape(inputs, list(input_shape[:-1]) + [group, input_shape[-1] // group])
        
        # operates on all dims except the group dim
        axes = list(range(1, input_shape.ndims-1)) + [input_shape.ndims]
        
        # create the shape of trainable variables
        shape = [1]*(len(range(input_shape.ndims-1))) + [input_shape[-1]]
        
        # create trainable variables
        beta = tf.get_variable(
            'beta',
            shape=shape,
            initializer=self.initializers['beta'],
            regularizer=self.regularizers['beta'],
            dtype=tf.float32, trainable=True)
        
        gamma = tf.get_variable(
            'gamma',
            shape=shape,
            initializer=self.initializers['gamma'],
            regularizer=self.regularizers['gamma'],
            dtype=tf.float32, trainable=True)
        
        # mean and var
        mean, variance = tf.nn.moments(inputs, axes, keep_dims=True)
        
        outputs = (inputs - mean) / tf.sqrt(variance + self.eps)
        
        outputs = tf.reshape(outputs, list(input_shape)) * gamma + beta

        return outputs