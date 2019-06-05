# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import math
import numpy as np
import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer, InstanceNormLayer
from niftynet.layer.gn import GNLayer
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_PADDING = set(['SAME', 'VALID', 'REFLECT', 'SYMMETRIC', 'CONSTANT'])


def default_w_initializer():
    def _initializer(shape, dtype, partition_info):
        stddev = np.sqrt(2.0 / np.prod(shape[:-1]))
        from tensorflow.python.ops import random_ops
        return random_ops.truncated_normal(shape, 0.0, stddev, dtype=tf.float32)
        # return tf.truncated_normal_initializer(
        #    mean=0.0, stddev=stddev, dtype=tf.float32)

    return _initializer


def default_b_initializer():
    return tf.constant_initializer(0.0)


class ConvLayer(TrainableLayer):
    """
    This class defines a simple convolution with an optional bias term.
    Please consider ``ConvolutionalLayer`` if batch_norm and activation
    are also used.
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 with_bias=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 padding_constant=0,
                 name='conv'):
        """
        :param padding_constant: a constant applied in padded convolution
        (see also tf.pad)
        """

        super(ConvLayer, self).__init__(name=name)

        self.padding = look_up_operations(padding.upper(), SUPPORTED_PADDING)
        self.n_output_chns = int(n_output_chns)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.with_bias = with_bias
        self.padding_constant = padding_constant

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor):
        input_shape = input_tensor.shape.as_list()
        n_input_chns = input_shape[-1]
        spatial_rank = layer_util.infer_spatial_rank(input_tensor)

        # initialize conv kernels/strides and then apply
        w_full_size = layer_util.expand_spatial_params(
            self.kernel_size, spatial_rank)
        # expand kernel size to include number of features
        w_full_size = w_full_size + (n_input_chns, self.n_output_chns)
        full_stride = layer_util.expand_spatial_params(
            self.stride, spatial_rank)
        full_dilation = layer_util.expand_spatial_params(
            self.dilation, spatial_rank)

        conv_kernel = tf.get_variable(
            'w', shape=w_full_size,
            initializer=self.initializers['w'],
            regularizer=self.regularizers['w'])
        if self.padding in ('VALID', 'SAME'):
            output_tensor = tf.nn.convolution(input=input_tensor,
                                              filter=conv_kernel,
                                              strides=full_stride,
                                              dilation_rate=full_dilation,
                                              padding=self.padding,
                                              name='conv')
        else:
            output_tensor = _extended_convolution(
                input_tensor,
                conv_kernel,
                full_stride,
                full_dilation,
                self.padding,
                constant=self.padding_constant)

        if not self.with_bias:
            return output_tensor

        # adding the bias term
        bias_term = tf.get_variable(
            'b', shape=self.n_output_chns,
            initializer=self.initializers['b'],
            regularizer=self.regularizers['b'])
        output_tensor = tf.nn.bias_add(output_tensor,
                                       bias_term,
                                       name='add_bias')
        return output_tensor


class ConvolutionalLayer(TrainableLayer):
    """
    This class defines a composite layer with optional components::

        convolution -> feature_normalization (default batch norm) -> activation -> dropout

    The b_initializer and b_regularizer are applied to the ConvLayer
    The w_initializer and w_regularizer are applied to the ConvLayer,
    the feature normalization layer, and the activation layer (for 'prelu')
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 with_bias=False,
                 feature_normalization='batch',
                 group_size=-1,
                 acti_func=None,
                 preactivation=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 padding_constant=0,
                 name="conv"):
        """
        :param padding_constant: constant applied with CONSTANT padding
        """

        self.acti_func = acti_func
        self.feature_normalization = feature_normalization
        self.group_size = group_size
        self.preactivation = preactivation
        self.layer_name = '{}'.format(name)
        if self.feature_normalization != 'group' and group_size > 0:
            raise ValueError('You cannot have a group_size > 0 if not using group norm')
        elif self.feature_normalization == 'group' and group_size <= 0:
            raise ValueError('You cannot have a group_size <= 0 if using group norm')

        if self.feature_normalization == 'batch':
            self.layer_name += '_bn'
        elif self.feature_normalization == 'group':
            self.layer_name += '_gn'
        elif self.feature_normalization == 'instance':
            self.layer_name += '_in'
        if self.acti_func is not None:
            self.layer_name += '_{}'.format(self.acti_func)
        super(ConvolutionalLayer, self).__init__(name=self.layer_name)

        # for ConvLayer
        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.with_bias = with_bias
        self.padding_constant = padding_constant

        # for BNLayer
        self.moving_decay = moving_decay
        self.eps = eps

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):
        conv_layer = ConvLayer(n_output_chns=self.n_output_chns,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               dilation=self.dilation,
                               padding=self.padding,
                               with_bias=self.with_bias,
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               b_initializer=self.initializers['b'],
                               b_regularizer=self.regularizers['b'],
                               padding_constant=self.padding_constant,
                               name='conv_')

        if self.feature_normalization == 'batch':
            if is_training is None:
                raise ValueError('is_training argument should be '
                                 'True or False unless feature_normalization is False')
            bn_layer = BNLayer(
                regularizer=self.regularizers['w'],
                moving_decay=self.moving_decay,
                eps=self.eps,
                name='bn_')
        elif self.feature_normalization == 'instance':
            in_layer = InstanceNormLayer(eps=self.eps, name='in_')
        elif self.feature_normalization == 'group':
            gn_layer = GNLayer(
                regularizer=self.regularizers['w'],
                group_size=self.group_size,
                eps=self.eps,
                name='gn_')
        if self.acti_func is not None:
            acti_layer = ActiLayer(
                func=self.acti_func,
                regularizer=self.regularizers['w'],
                name='acti_')

        if keep_prob is not None:
            dropout_layer = ActiLayer(func='dropout', name='dropout_')

        def activation(output_tensor):
            if self.feature_normalization == 'batch':
                output_tensor = bn_layer(output_tensor, is_training)
            elif self.feature_normalization == 'instance':
                output_tensor = in_layer(output_tensor)
            elif self.feature_normalization == 'group':
                output_tensor = gn_layer(output_tensor)
            if self.acti_func is not None:
                output_tensor = acti_layer(output_tensor)
            if keep_prob is not None:
                output_tensor = dropout_layer(output_tensor,
                                              keep_prob=keep_prob)
            return output_tensor

        if self.preactivation:
            output_tensor = conv_layer(activation(input_tensor))
        else:
            output_tensor = activation(conv_layer(input_tensor))

        return output_tensor


def _compute_pad_size(input_dim_size, output_dim_size, kernel_dim_size,
                      stride, dilation):
    """
    Computes the size of the pad using the formula given in TF's conv_ops.cc.
    :return: the one-sided pad size
    """

    return ((output_dim_size - 1)*stride + (kernel_dim_size - 1)*dilation + 2
            - input_dim_size)//2


def _extended_convolution(input_tensor,
                          kernel,
                          strides,
                          dilations,
                          padding,
                          constant=0,
                          name='extended_convolution'):
    """
    A simple wrapper for tf.nn.convolution that first expands the input tensor
    by sampling at discrete locations in the original tensor then invokes
    the original convolution operation on the expanded tensor, and finally
    extracts a suitable output tensor from the output of the convolution of
    the expanded tensor.
    :param input_tensor: original convolution input tensor
    :param kernel: convolution kernel
    :param strides: strided convolution strides (one per spatial dimension)
    :param dilations: dilated convolution dilation factors
    (one per spatial dimension)
    :param padding: a string specifying the type of padding to apply
    :param constant: a padding constant (only read in the case of constant
    padding)
    :param name: a name for the operation
    :return: a convolution result of the same size as the input tensor
    """

    input_shape = input_tensor.shape.as_list()
    batch_size = input_shape[0]
    input_shape = input_shape[1:-1]
    kernel_shape = kernel.shape.as_list()
    nof_output_features = kernel_shape[-1]
    kernel_shape = kernel_shape[:-2]

    if any(i is None or i < 0 or k is None or k < 0
           for i, k in zip(input_shape, kernel_shape)):
        raise ValueError('The dimensions of the input tensor and the filter'
                         ' must be known in advance for this operation to '
                         'work.')

    output_shape = [int(math.ceil(i/s)) for i, s in zip(input_shape, strides)]
    output_shape = [batch_size] + output_shape + [nof_output_features]

    dimpads = [0]
    for i, k, s, d in zip(input_shape, kernel_shape, strides, dilations):
        pad = _compute_pad_size(i, int(math.ceil(i/s)), k, s, d)
        dimpads.append(pad)
    dimpads += [0]

    # Cannot pad by more than 1 dimension size => repeatedly pad
    if padding in ('REFLECT', 'SYMMETRIC'):
        padded_input = input_tensor
        offset = int(padding == 'REFLECT')

        while min(o - i - 2*p for o, i, p in zip(
                padded_input.shape.as_list()[1:-1],
                input_shape,
                dimpads[1:-1])) < 0:
            effective_pad = [(0, 0)]
            padded_shape = padded_input.shape.as_list()[1:-1]
            for i in range(len(input_shape)):
                epad = min((input_shape[i] + 2*dimpads[1+i] - padded_shape[i])//2,
                           padded_shape[i] - offset)
                epad = max(epad, 0)
                effective_pad.append((epad, epad))
            effective_pad += [(0, 0)]

            assert max(e for e, _ in effective_pad) > 0

            padded_input = tf.pad(padded_input,
                                  effective_pad,
                                  mode=padding)
    else:
        padded_input = tf.pad(input_tensor,
                              [(d, d) for d in dimpads],
                              mode=padding,
                              constant_values=constant)

    conv_output = tf.nn.convolution(input=padded_input,
                                    filter=kernel,
                                    strides=strides,
                                    dilation_rate=dilations,
                                    padding='SAME',
                                    name='conv_' + name)

    conv_output_shape = conv_output.shape.as_list()
    out_pad = [0]
    out_pad += [(o - i)//2 for i, o in zip(output_shape[1:-1], conv_output_shape[1:-1])]
    out_pad += [0]

    return tf.slice(conv_output, out_pad, output_shape) if max(out_pad) > 0 \
        else conv_output
