# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

# from utilities.misc_common import look_up_operations
# from . import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer, InstanceNormLayer
from niftynet.layer.gn import GNLayer


def default_w_initializer():
    def _initializer(shape, dtype, partition_info):
        stddev = np.sqrt(2.0 / np.prod(shape[:-1]))
        from tensorflow.python.ops import random_ops
        return random_ops.truncated_normal(
            shape, 0.0, stddev, dtype=tf.float32)
        # return tf.truncated_normal_initializer(
        #    mean=0.0, stddev=stddev, dtype=tf.float32)

    return _initializer


def default_b_initializer():
    return tf.constant_initializer(0.0)


class FCLayer(TrainableLayer):
    """
    This class defines a simple fully connected layer with
    an optional bias term.
    Please consider ``FullyConnectedLayer`` if batch_norm and activation
    are also used.
    """

    def __init__(self,
                 n_output_chns,
                 with_bias=True,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='fc'):
        super(FCLayer, self).__init__(name=name)

        self.n_output_chns = n_output_chns
        self.with_bias = with_bias

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor):
        input_shape = input_tensor.shape.as_list()
        if len(input_shape) > 2:
            batch_size = input_shape[0]
            input_tensor = tf.reshape(input_tensor, [batch_size, -1])
            input_shape = input_tensor.shape.as_list()
        n_input_chns = input_shape[-1]

        # initialize weight matrix and then apply
        weight_matrix = tf.get_variable(
            'w', shape=[n_input_chns, self.n_output_chns],
            initializer=self.initializers['w'],
            regularizer=self.regularizers['w'])
        output_tensor = tf.matmul(input_tensor,
                                  weight_matrix,
                                  name='fc')
        if not self.with_bias:
            return output_tensor

        # adding the bias term
        bias_term = tf.get_variable(
            'b', shape=self.n_output_chns,
            initializer=self.initializers['b'],
            regularizer=self.regularizers['b'])
        output_tensor = tf.nn.bias_add(
            output_tensor, bias_term, name='add_bias')
        return output_tensor


class FullyConnectedLayer(TrainableLayer):
    """
    This class defines a composite layer with optional components::

        fully connected layer -> batch_norm -> activation -> dropout

    The b_initializer and b_regularizer are applied to the FCLayer
    The w_initializer and w_regularizer are applied to the FCLayer,
    the batch normalisation layer, and the activation layer (for 'prelu')
    """

    def __init__(self,
                 n_output_chns,
                 with_bias=True,
                 feature_normalization='batch',
                 group_size=-1,
                 acti_func=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 name="fc"):

        self.acti_func = acti_func
        self.feature_normalization = feature_normalization
        self.group_size = group_size
        self.layer_name = '{}'.format(name)

        if self.feature_normalization != 'group' and group_size > 0:
            raise ValueError('You cannot have a group_size > 0 if not using group norm')
        elif self.feature_normalization == 'group' and group_size <= 0:
            raise ValueError('You cannot have a group_size <= 0 if using group norm')

        if self.feature_normalization is not None:
            # to append batch_norm as _bn and likewise for other norms
            self.layer_name += '_' + self.feature_normalization[0] + 'n'
        if self.acti_func is not None:
            self.layer_name += '_{}'.format(self.acti_func)
        super(FullyConnectedLayer, self).__init__(name=self.layer_name)

        # for FCLayer
        self.n_output_chns = n_output_chns
        self.with_bias = with_bias

        # for BNLayer
        self.moving_decay = moving_decay
        self.eps = eps

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):
        fc_layer = FCLayer(n_output_chns=self.n_output_chns,
                           with_bias=self.with_bias,
                           w_initializer=self.initializers['w'],
                           w_regularizer=self.regularizers['w'],
                           b_initializer=self.initializers['b'],
                           b_regularizer=self.regularizers['b'],
                           name='fc_')
        output_tensor = fc_layer(input_tensor)

        if self.feature_normalization == 'batch':
            if is_training is None:
                raise ValueError('is_training argument should be '
                                 'True or False unless feature_normalization is False')
            bn_layer = BNLayer(
                regularizer=self.regularizers['w'],
                moving_decay=self.moving_decay,
                eps=self.eps,
                name='bn_')
            output_tensor = bn_layer(output_tensor, is_training)
        elif self.feature_normalization == 'instance':
            in_layer = InstanceNormLayer(eps=self.eps, name='in_')
            output_tensor = in_layer(output_tensor)
        elif self.feature_normalization == 'group':
            gn_layer = GNLayer(
                regularizer=self.regularizers['w'],
                group_size=self.group_size,
                eps=self.eps,
                name='gn_')
            output_tensor = gn_layer(output_tensor)

        if self.acti_func is not None:
            acti_layer = ActiLayer(
                func=self.acti_func,
                regularizer=self.regularizers['w'],
                name='acti_')
            output_tensor = acti_layer(output_tensor)

        if keep_prob is not None:
            dropout_layer = ActiLayer(func='dropout', name='dropout_')
            output_tensor = dropout_layer(output_tensor, keep_prob=keep_prob)

        return output_tensor
