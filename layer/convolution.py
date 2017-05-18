import numpy as np
import tensorflow as tf

from . import layer_util
from .activation import ActiLayer
from .base import TrainableLayer
from .bn import BNLayer

SUPPORTED_PADDING = set(['SAME', 'VALID'])


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
    Please consider `ConvolutionalLayer` if batch_norm and activation
    are also used.
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 padding='SAME',
                 w_initializer=None,
                 w_regularizer=None,
                 with_bias=False,
                 b_initializer=None,
                 b_regularizer=None,
                 name='conv'):
        self.padding = padding.upper()
        assert self.padding in SUPPORTED_PADDING

        self.layer_name = '{}'.format(name)
        super(ConvLayer, self).__init__(name=self.layer_name)

        self.n_output_chns = n_output_chns
        self.kernel_size = np.asarray(kernel_size).flatten()
        self.stride = np.asarray(stride).flatten()
        self.with_bias = with_bias

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor):
        input_shape = input_tensor.get_shape().as_list()
        n_input_chns = input_shape[-1]
        spatial_rank = layer_util.infer_spatial_rank(input_tensor)

        # initialize conv kernels/strides and then apply
        w_full_size = np.vstack((
            [self.kernel_size] * spatial_rank,
            n_input_chns, self.n_output_chns)).flatten()
        full_stride = np.vstack((
            [self.stride] * spatial_rank)).flatten()
        conv_kernel = tf.get_variable(
            'w', shape=w_full_size.tolist(),
            initializer=self.initializers['w'],
            regularizer=self.regularizers['w'])
        output_tensor = tf.nn.convolution(input=input_tensor,
                                          filter=conv_kernel,
                                          strides=full_stride.tolist(),
                                          padding=self.padding,
                                          name='conv')
        if not self.with_bias:
            return output_tensor

        # adding the bias term
        bias_term = tf.get_variable(
            'b', shape=(self.n_output_chns),
            initializer=self.initializers['b'],
            regularizer=self.regularizers['b'])
        output_tensor = tf.nn.bias_add(output_tensor, bias_term, name='add_bias')
        return output_tensor


class ConvolutionalLayer(TrainableLayer):
    """
    This class defines a composite layer with optional components:
        convolution -> batch_norm -> activation -> dropout
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size,
                 stride=1,
                 padding='SAME',
                 w_initializer=None,
                 w_regularizer=None,
                 with_bias=False,
                 b_initializer=None,
                 b_regularizer=None,
                 with_bn=True,
                 bn_regularizer=None,
                 acti_fun=None,
                 name="conv"):

        self.with_bias = with_bias
        self.acti_fun = acti_fun
        self.with_bn = with_bn
        self.layer_name = '{}'.format(name)
        if self.with_bn:
            self.layer_name += '_bn'
        if (self.acti_fun is not None):
            self.layer_name += '_{}'.format(self.acti_fun)
        super(ConvolutionalLayer, self).__init__(name=self.layer_name)

        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        self.conv_layer = None
        self.bn_layer = None
        self.acti_layer = None
        self.dropout_layer = None

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):
        # init sub-layers
        self.conv_layer = ConvLayer(n_output_chns=self.n_output_chns,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    w_initializer=self.initializers['w'],
                                    w_regularizer=self.regularizers['w'],
                                    with_bias=self.with_bias,
                                    b_initializer=self.initializers['b'],
                                    b_regularizer=self.regularizers['b'],
                                    name='conv_')
        output_tensor = self.conv_layer(input_tensor)

        if self.with_bn:
            self.bn_layer = BNLayer(regularizer=self.regularizers['w'],
                                    name='bn_')
            output_tensor = self.bn_layer(output_tensor, is_training)

        if self.acti_fun is not None:
            self.acti_layer = ActiLayer(func=self.acti_fun,
                                        regularizer=self.regularizers['w'],
                                        name='acti_')
            output_tensor = self.acti_layer(output_tensor)

        if keep_prob is not None:
            self.dropout_layer = ActiLayer(func='dropout',
                                           name='dropout_')
            output_tensor = self.dropout_layer(output_tensor,
                                               keep_prob=keep_prob)
        return output_tensor
