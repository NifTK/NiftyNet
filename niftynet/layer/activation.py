# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer
from niftynet.utilities.util_common import look_up_operations


def prelu(f_in, channelwise_params):
    pos = tf.nn.relu(f_in)
    neg = channelwise_params * (f_in - tf.abs(f_in)) * 0.5
    return pos + neg


def selu(x, name):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def leaky_relu(x, name):
    half_alpha = 0.01
    return (0.5 + half_alpha) * x + (0.5 - half_alpha) * abs(x)


SUPPORTED_OP = {'relu': tf.nn.relu,
                'relu6': tf.nn.relu6,
                'elu': tf.nn.elu,
                'softplus': tf.nn.softplus,
                'softsign': tf.nn.softsign,
                'sigmoid': tf.nn.sigmoid,
                'tanh': tf.nn.tanh,
                'prelu': prelu,
                'selu': selu,
                'leakyrelu': leaky_relu,
                'dropout': tf.nn.dropout}


class ActiLayer(TrainableLayer):
    """
    Apply an element-wise non-linear activation function.
    'Prelu' uses trainable parameters and those are initialised to zeros
    Dropout function is also supported
    """

    def __init__(self, func, regularizer=None, name='activation'):
        self.func = func.lower()
        self.layer_name = '{}_{}'.format(self.func, name)

        super(ActiLayer, self).__init__(name=self.layer_name)

        # these are used for prelu variables
        self.initializers = {'alpha': tf.constant_initializer(0.0)}
        self.regularizers = {'alpha': regularizer}

    def layer_op(self, input_tensor, keep_prob=None):
        func_ = look_up_operations(self.func, SUPPORTED_OP)
        if self.func == 'prelu':
            alphas = tf.get_variable(
                'alpha', input_tensor.shape[-1],
                initializer=self.initializers['alpha'],
                regularizer=self.regularizers['alpha'])
            output_tensor = func_(input_tensor, alphas)
        elif self.func == 'dropout':
            output_tensor = func_(input_tensor,
                                  keep_prob=keep_prob,
                                  name='dropout')
        else:
            output_tensor = func_(input_tensor, name='acti')
        return output_tensor
