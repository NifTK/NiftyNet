import numpy as np
import tensorflow as tf

from . import layer_util
from .activation import ActiLayer
from .base import TrainableLayer
from .bn import BNLayer

SUPPORTED_OP = {'2D': tf.nn.conv2d_transpose,
                '3D': tf.nn.conv3d_transpose}
SUPPORTED_PADDING = {'SAME', 'VALID'}


def default_w_initializer():
    def _initializer(shape, dtype, partition_info):
        stddev = np.sqrt(2.0 / np.prod(shape[:-2]) * shape[-1])
        from tensorflow.python.ops import random_ops
        return random_ops.truncated_normal(shape, 0.0, stddev, dtype=tf.float32)
        # return tf.truncated_normal_initializer(
        #    mean=0.0, stddev=stddev, dtype=tf.float32)

    return _initializer


def default_b_initializer():
    return tf.constant_initializer(0.0)


def infer_output_dim(input_dim, stride, kernel_size, padding):
    assert input_dim is not None
    if padding == 'VALID':
        output_dim = input_dim * stride + max(kernel_size - stride, 0)
    if padding == 'SAME':
        output_dim = input_dim * stride
    return output_dim


class DeconvLayer(TrainableLayer):
    """
    This class defines a simple deconvolution with an optional bias term.
    Please consider `DeconvolutionalLayer` if batch_norm and activation
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
        self.padding = padding.upper()
        assert self.padding in SUPPORTED_PADDING

        self.layer_name = '{}'.format(name)
        super(DeconvLayer, self).__init__(name=self.layer_name)

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
            self.n_output_chns, n_input_chns)).flatten()
        full_stride = np.vstack((
            1, [self.stride] * spatial_rank, 1)).flatten()
        deconv_kernel = tf.get_variable(
            'w', shape=w_full_size.tolist(),
            initializer=self.initializers['w'],
            regularizer=self.regularizers['w'])
        if spatial_rank == 2:
            op_ = SUPPORTED_OP['2D']
        elif spatial_rank == 3:
            op_ = SUPPORTED_OP['3D']
        else:
            raise ValueError(
                "Only 2D and 3D spatial deconvolutions are supported")

        output_dim = infer_output_dim(input_shape[1],
                                      self.stride,
                                      self.kernel_size,
                                      self.padding)
        full_output_size = np.vstack((input_shape[0],
                                      [output_dim] * spatial_rank,
                                      self.n_output_chns)).flatten()
        output_tensor = op_(value=input_tensor,
                            filter=deconv_kernel,
                            output_shape=full_output_size.tolist(),
                            strides=full_stride.tolist(),
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
    This class defines a composite layer with optional components:
        deconvolution -> batch_norm -> activation -> dropout
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size,
                 stride,
                 padding='SAME',
                 with_bias=False,
                 with_bn=True,
                 acti_func=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name="deconv"):

        self.with_bias = with_bias
        self.acti_func = acti_func
        self.with_bn = with_bn
        self.layer_name = '{}'.format(name)
        # if self.with_bn:
        #    self.layer_name += '_bn'
        # if (self.acti_func is not None):
        #    self.layer_name += '_{}'.format(self.acti_func)
        super(DeconvolutionalLayer, self).__init__(name=self.layer_name)

        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        self.deconv_layer = None
        self.bn_layer = None
        self.acti_layer = None
        self.dropout_layer = None

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):
        # init sub-layers
        self.deconv_layer = DeconvLayer(n_output_chns=self.n_output_chns,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=self.padding,
                                        with_bias=self.with_bias,
                                        w_initializer=self.initializers['w'],
                                        w_regularizer=self.regularizers['w'],
                                        b_initializer=self.initializers['b'],
                                        b_regularizer=self.regularizers['b'],
                                        name='deconv_')
        output_tensor = self.deconv_layer(input_tensor)

        if self.with_bn:
            self.bn_layer = BNLayer(regularizer=self.regularizers['w'],
                                    name='bn_')
            output_tensor = self.bn_layer(output_tensor, is_training)

        if self.acti_func is not None:
            self.acti_layer = ActiLayer(func=self.acti_func,
                                        regularizer=self.regularizers['w'],
                                        name='acti_')
            output_tensor = self.acti_layer(output_tensor)

        if keep_prob is not None:
            self.dropout_layer = ActiLayer(func='dropout', name='dropout_')
            output_tensor = self.dropout_layer(output_tensor,
                                               keep_prob=keep_prob)
        return output_tensor
