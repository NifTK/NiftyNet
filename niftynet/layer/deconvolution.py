# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer, InstanceNormLayer
from niftynet.layer.gn import GNLayer
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_OP = {
    '2D': tf.nn.conv2d_transpose,
    '3D': tf.nn.conv3d_transpose}
SUPPORTED_PADDING = set(['SAME', 'VALID'])


def default_w_initializer():
    def _initializer(shape, dtype, partition_info):
        stddev = np.sqrt(2.0 / (np.prod(shape[:-2]) * shape[-1]))
        from tensorflow.python.ops import random_ops
        return random_ops.truncated_normal(shape, 0.0, stddev, dtype=tf.float32)
        # return tf.truncated_normal_initializer(
        #    mean=0.0, stddev=stddev, dtype=tf.float32)

    return _initializer


def default_b_initializer():
    return tf.constant_initializer(0.0)


def infer_output_dims(input_dims, strides, kernel_sizes, padding):
    """
    infer output dims from list,
    the dim can be different in different directions.
    Note: dilation is not considered here.
    """
    assert len(input_dims) == len(strides)
    assert len(input_dims) == len(kernel_sizes)
    output_dims = []
    for (i, dim) in enumerate(input_dims):
        if dim is None:
            output_dims.append(None)
            continue
        if padding == 'VALID':
            output_dims.append(
                dim * strides[i] + max(kernel_sizes[i] - strides[i], 0))
        else:
            output_dims.append(dim * strides[i])
    return output_dims


class DeconvLayer(TrainableLayer):
    """
    This class defines a simple deconvolution with an optional bias term.
    Please consider ``DeconvolutionalLayer`` if batch_norm and activation
    are also used.
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 padding='SAME',
                 with_bias=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='deconv'):

        super(DeconvLayer, self).__init__(name=name)

        self.padding = look_up_operations(padding.upper(), SUPPORTED_PADDING)
        self.n_output_chns = int(n_output_chns)
        self.kernel_size = kernel_size
        self.stride = stride
        self.with_bias = with_bias

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor):
        input_shape = input_tensor.shape.as_list()
        n_input_chns = input_shape[-1]
        spatial_rank = layer_util.infer_spatial_rank(input_tensor)

        # initialize conv kernels/strides and then apply
        kernel_size_all_dim = layer_util.expand_spatial_params(
            self.kernel_size, spatial_rank)
        w_full_size = kernel_size_all_dim + (self.n_output_chns, n_input_chns)
        stride_all_dim = layer_util.expand_spatial_params(
            self.stride, spatial_rank)
        full_stride = (1,) + stride_all_dim + (1,)

        deconv_kernel = tf.get_variable(
            'w', shape=w_full_size,
            initializer=self.initializers['w'],
            regularizer=self.regularizers['w'])
        if spatial_rank == 2:
            op_ = SUPPORTED_OP['2D']
        elif spatial_rank == 3:
            op_ = SUPPORTED_OP['3D']
        else:
            raise ValueError(
                "Only 2D and 3D spatial deconvolutions are supported")

        spatial_shape = []
        for (i, dim) in enumerate(input_shape[:-1]):
            if i == 0:
                continue
            if dim is None:
                spatial_shape.append(tf.shape(input_tensor)[i])
            else:
                spatial_shape.append(dim)
        output_dims = infer_output_dims(spatial_shape,
                                        stride_all_dim,
                                        kernel_size_all_dim,
                                        self.padding)
        if input_tensor.shape.is_fully_defined():
            full_output_size = \
                [input_shape[0]] + output_dims + [self.n_output_chns]
        else:
            batch_size = tf.shape(input_tensor)[0]
            full_output_size = tf.stack(
                [batch_size] + output_dims + [self.n_output_chns])
        output_tensor = op_(value=input_tensor,
                            filter=deconv_kernel,
                            output_shape=full_output_size,
                            strides=full_stride,
                            padding=self.padding,
                            name='deconv')
        if not self.with_bias:
            return output_tensor

        # adding the bias term
        bias_full_size = (self.n_output_chns,)
        bias_term = tf.get_variable(
            'b', shape=bias_full_size,
            initializer=self.initializers['b'],
            regularizer=self.regularizers['b'])
        output_tensor = tf.nn.bias_add(output_tensor,
                                       bias_term,
                                       name='add_bias')
        return output_tensor


class DeconvolutionalLayer(TrainableLayer):
    """
    This class defines a composite layer with optional components::

        deconvolution -> batch_norm -> activation -> dropout

    The b_initializer and b_regularizer are applied to the DeconvLayer
    The w_initializer and w_regularizer are applied to the DeconvLayer,
    the batch normalisation layer, and the activation layer (for 'prelu')
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 padding='SAME',
                 with_bias=False,
                 feature_normalization='batch',
                 group_size=-1,
                 acti_func=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 name="deconv"):

        self.acti_func = acti_func
        self.feature_normalization = feature_normalization
        self.group_size = group_size
        self.layer_name = '{}'.format(name)
        if self.feature_normalization != 'group' and group_size > 0:
            raise ValueError('You cannot have a group_size > 0 if not using group norm')
        elif self.feature_normalization == 'group' and group_size <= 0:
            raise ValueError('You cannot have a group_size <= 0 if using group norm')

        if self.feature_normalization is not None:
            # appending, for example, '_bn' to the name 
            self.layer_name += '_' + self.feature_normalization[0] + 'n'
        if self.acti_func is not None:
            self.layer_name += '_{}'.format(self.acti_func)
        super(DeconvolutionalLayer, self).__init__(name=self.layer_name)

        # for DeconvLayer
        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.with_bias = with_bias

        # for BNLayer
        self.moving_decay = moving_decay
        self.eps = eps

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):
        # init sub-layers
        deconv_layer = DeconvLayer(n_output_chns=self.n_output_chns,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding,
                                   with_bias=self.with_bias,
                                   w_initializer=self.initializers['w'],
                                   w_regularizer=self.regularizers['w'],
                                   b_initializer=self.initializers['b'],
                                   b_regularizer=self.regularizers['b'],
                                   name='deconv_')
        output_tensor = deconv_layer(input_tensor)

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
