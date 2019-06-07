# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import math

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

import niftynet.layer.bn
import niftynet.layer.convolution
import niftynet.layer.deconvolution
from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.deconvolution import infer_output_dims

SUPPORTED_OP = {'2D': tf.nn.conv2d_transpose,
                '3D': tf.nn.conv3d_transpose}


class ChannelSparseDeconvLayer(niftynet.layer.deconvolution.DeconvLayer):
    """
    Channel sparse convolutions perform convolutions over
    a subset of image channels and generate a subset of output
    channels. This enables spatial dropout without wasted computations
    """

    def __init__(self, *args, **kwargs):
        super(ChannelSparseDeconvLayer, self).__init__(*args, **kwargs)

    def layer_op(self, input_tensor, input_mask=None, output_mask=None):
        """

        :param input_tensor: image to convolve with kernel
        :param input_mask: 1-Tensor with a binary mask of input channels to use
            If this is None, all channels are used.
        :param output_mask: 1-Tensor with a binary mask of output channels to
            generate. If this is None, all channels are used and the number
            of output channels is set at graph-creation time.
        :return:
        """

        input_shape = input_tensor.shape.as_list()
        if input_mask is None:
            _input_mask = tf.ones([input_shape[-1]]) > 0
        else:
            _input_mask = input_mask

        if output_mask is None:
            n_sparse_output_chns = self.n_output_chns
            _output_mask = tf.ones([self.n_output_chns]) > 0
        else:
            n_sparse_output_chns = tf.reduce_sum(
                tf.cast(output_mask, tf.float32))
            _output_mask = output_mask

        n_full_input_chns = _input_mask.shape.as_list()[0]
        spatial_rank = layer_util.infer_spatial_rank(input_tensor)

        # initialize conv kernels/strides and then apply
        w_full_size = np.vstack((
            [self.kernel_size] * spatial_rank,
            self.n_output_chns, n_full_input_chns)).flatten()
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

        output_dim = infer_output_dims(input_shape[1],
                                       self.stride,
                                       self.kernel_size,
                                       self.padding)
        sparse_output_size = \
            [input_shape[0], [output_dim] * spatial_rank, n_sparse_output_chns]
        sparse_output_size = tf.stack(sparse_output_size, 0)
        output_tensor = op_(value=input_tensor,
                            filter=deconv_kernel,
                            output_shape=sparse_output_size,
                            strides=full_stride.tolist(),
                            padding=self.padding,
                            name='deconv')
        if output_mask is None:
            # If all output channels are used, we can specify
            # the number of output channels which is useful for later layers
            old_shape = output_tensor.shape.as_list()
            old_shape[-1] = self.n_output_chns
            output_tensor.set_shape(old_shape)
        if not self.with_bias:
            return output_tensor

        # adding the bias term
        bias_full_size = (self.n_output_chns,)
        bias_term = tf.get_variable(
            'b', shape=bias_full_size,
            initializer=self.initializers['b'],
            regularizer=self.regularizers['b'])
        sparse_bias = tf.boolean_mask(bias_term, _output_mask)

        output_tensor = tf.nn.bias_add(
            output_tensor, sparse_bias, name='add_bias')
        return output_tensor


class ChannelSparseConvLayer(niftynet.layer.convolution.ConvLayer):
    """
    Channel sparse convolutions perform convolutions over
    a subset of image channels and generate a subset of output
    channels. This enables spatial dropout without wasted computations.
    """

    def __init__(self, *args, **kwargs):
        super(ChannelSparseConvLayer, self).__init__(*args, **kwargs)

    def layer_op(self, input_tensor, input_mask, output_mask):
        """

        :param input_tensor: image to convolve with kernel
        :param input_mask: 1-Tensor with a binary mask of input channels to use
            If this is None, all channels are used.
        :param output_mask: 1-Tensor with a binary mask of output channels to
            generate. If this is None, all channels are used and
            the number of output channels is set at graph-creation time.
        :return:
        """
        sparse_input_shape = input_tensor.shape.as_list()
        if input_mask is None:
            _input_mask = tf.ones([sparse_input_shape[-1]]) > 0
        else:
            _input_mask = input_mask
        if output_mask is None:
            _output_mask = tf.ones([self.n_output_chns]) > 0
        else:
            _output_mask = output_mask
        n_full_input_chns = _input_mask.shape.as_list()[0]
        spatial_rank = layer_util.infer_spatial_rank(input_tensor)
        # initialize conv kernels/strides and then apply
        w_full_size = layer_util.expand_spatial_params(
            self.kernel_size, spatial_rank)
        # expand kernel size to include number of features
        w_full_size = w_full_size + (n_full_input_chns, self.n_output_chns)

        full_stride = layer_util.expand_spatial_params(
            self.stride, spatial_rank)

        full_dilation = layer_util.expand_spatial_params(
            self.dilation, spatial_rank)

        conv_kernel = tf.get_variable(
            'w', shape=w_full_size,
            initializer=self.initializers['w'],
            regularizer=self.regularizers['w'])

        if spatial_rank == 2:
            transpositions = [[3, 2, 1, 0], [1, 0, 2, 3], [3, 2, 0, 1]]
        elif spatial_rank == 3:
            transpositions = [[4, 3, 2, 1, 0], [1, 0, 2, 3, 4], [4, 3, 2, 0, 1]]
        else:
            raise NotImplementedError("spatial rank not supported")

        sparse_kernel = tf.transpose(conv_kernel, transpositions[0])
        sparse_kernel = tf.boolean_mask(sparse_kernel, _output_mask)
        sparse_kernel = tf.transpose(sparse_kernel, transpositions[1])
        sparse_kernel = tf.boolean_mask(sparse_kernel, _input_mask)
        sparse_kernel = tf.transpose(sparse_kernel, transpositions[2])

        output_tensor = tf.nn.convolution(input=input_tensor,
                                          filter=sparse_kernel,
                                          strides=full_stride,
                                          dilation_rate=full_dilation,
                                          padding=self.padding,
                                          name='conv')
        if output_mask is None:
            # If all output channels are used, we can specify
            # the number of output channels which is useful for later layers
            old_shape = output_tensor.shape.as_list()
            old_shape[-1] = self.n_output_chns
            output_tensor.set_shape(old_shape)

        if not self.with_bias:
            return output_tensor

        # adding the bias term
        bias_term = tf.get_variable(
            'b', shape=self.n_output_chns,
            initializer=self.initializers['b'],
            regularizer=self.regularizers['b'])
        sparse_bias = tf.boolean_mask(bias_term, output_mask)
        output_tensor = tf.nn.bias_add(
            output_tensor, sparse_bias, name='add_bias')
        return output_tensor


class ChannelSparseBNLayer(niftynet.layer.bn.BNLayer):
    """
    Channel sparse convolutions perform convolutions over
    a subset of image channels and generate a subset of output
    channels. This enables spatial dropout without wasted computations
    """

    def __init__(self, n_dense_channels, *args, **kwargs):
        self.n_dense_channels = n_dense_channels
        super(ChannelSparseBNLayer, self).__init__(*args, **kwargs)

    def layer_op(self, inputs, is_training, mask, use_local_stats=False):
        """

        :param inputs: image to normalize. This typically represents a sparse
            subset of channels from a sparse convolution.
        :param is_training: boolean that is True during training.
            When True, the layer uses batch statistics for normalization and
            records a moving average of means and variances.
            When False, the layer uses previously computed moving averages
            for normalization.
        :param mask: 1-Tensor with a binary mask identifying the sparse
            channels represented in inputs
        :param use_local_stats:
        :return:
        """

        if mask is None:
            mask = tf.ones([self.n_dense_channels]) > 0
        else:
            mask = mask

        input_shape = inputs.shape
        mask_shape = mask.shape
        # operates on all dims except the last dim
        params_shape = mask_shape[-1:]
        assert params_shape[0] == self.n_dense_channels, \
            'Mask size {} must match n_dense_channels {}.'.format(
                params_shape[0], self.n_dense_channels)
        axes = list(range(input_shape.ndims - 1))
        # create trainable variables and moving average variables
        beta = tf.get_variable(
            'beta',
            shape=params_shape,
            initializer=self.initializers['beta'],
            regularizer=self.regularizers['beta'],
            dtype=tf.float32, trainable=True)
        gamma = tf.get_variable(
            'gamma',
            shape=params_shape,
            initializer=self.initializers['gamma'],
            regularizer=self.regularizers['gamma'],
            dtype=tf.float32, trainable=True)
        beta = tf.boolean_mask(beta, mask)
        gamma = tf.boolean_mask(gamma, mask)

        collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        moving_mean = tf.get_variable(
            'moving_mean',
            shape=params_shape,
            initializer=self.initializers['moving_mean'],
            dtype=tf.float32, trainable=False, collections=collections)
        moving_variance = tf.get_variable(
            'moving_variance',
            shape=params_shape,
            initializer=self.initializers['moving_variance'],
            dtype=tf.float32, trainable=False, collections=collections)

        # mean and var
        mean, variance = tf.nn.moments(inputs, axes)
        # only update masked moving averages
        mean_update = tf.dynamic_stitch(
            [tf.to_int32(tf.where(mask)[:, 0]),
             tf.to_int32(tf.where(~mask)[:, 0])],
            [mean,
             tf.boolean_mask(moving_mean, ~mask)])
        variance_update = tf.dynamic_stitch(
            [tf.to_int32(tf.where(mask)[:, 0]),
             tf.to_int32(tf.where(~mask)[:, 0])],
            [variance,
             tf.boolean_mask(moving_variance, ~mask)])
        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean_update, self.moving_decay).op
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance_update, self.moving_decay).op
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

        # call the normalisation function
        if is_training or use_local_stats:
            outputs = tf.nn.batch_normalization(
                inputs, mean, variance,
                beta, gamma, self.eps, name='batch_norm')
        else:
            outputs = tf.nn.batch_normalization(
                inputs,
                tf.boolean_mask(moving_mean, mask),
                tf.boolean_mask(moving_variance, mask),
                beta, gamma, self.eps, name='batch_norm')
        outputs.set_shape(inputs.get_shape())
        return outputs


class ChannelSparseConvolutionalLayer(TrainableLayer):
    """
    This class defines a composite layer with optional components::

      channel sparse convolution ->
      batchwise-spatial dropout ->
      batch_norm ->
      activation

    The b_initializer and b_regularizer are applied to
    the ChannelSparseConvLayer, the w_initializer and w_regularizer
    are applied to the ChannelSparseConvLayer, the batch normalisation
    layer, and the activation layer (for 'prelu')
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 with_bias=False,
                 feature_normalization='batch',
                 acti_func=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 name="conv"):

        self.acti_func = acti_func
        self.feature_normalization = feature_normalization
        self.layer_name = '{}'.format(name)
        if self.feature_normalization == 'batch':
            self.layer_name += '_bn'
        if self.acti_func is not None:
            self.layer_name += '_{}'.format(self.acti_func)
        super(ChannelSparseConvolutionalLayer, self).__init__(
            name=self.layer_name)

        # for ConvLayer
        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.with_bias = with_bias

        # for BNLayer
        self.moving_decay = moving_decay
        self.eps = eps

        self.initializers = {
            'w': w_initializer if w_initializer else
            niftynet.layer.convolution.default_w_initializer(),
            'b': b_initializer if b_initializer else
            niftynet.layer.convolution.default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self,
                 input_tensor,
                 input_mask=None,
                 is_training=None,
                 keep_prob=None):
        conv_layer = ChannelSparseConvLayer(
            n_output_chns=self.n_output_chns,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            with_bias=self.with_bias,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            b_initializer=self.initializers['b'],
            b_regularizer=self.regularizers['b'],
            name='conv_')
        if keep_prob is not None:
            output_mask = \
                tf.to_float(tf.random_shuffle(tf.range(self.n_output_chns))) \
                < keep_prob * self.n_output_chns
            n_output_ch = math.ceil(keep_prob * self.n_output_chns)
        else:
            output_mask = tf.ones([self.n_output_chns]) > 0
            n_output_ch = self.n_output_chns

        output_tensor = conv_layer(input_tensor, input_mask, output_mask)
        output_tensor.set_shape(
            output_tensor.shape.as_list()[:-1] + [n_output_ch])

        if self.feature_normalization == 'batch':
            if is_training is None:
                raise ValueError('For batch norm, you must set the `is_training` argument.')
            bn_layer = ChannelSparseBNLayer(
                self.n_output_chns,
                regularizer=self.regularizers['w'],
                moving_decay=self.moving_decay,
                eps=self.eps,
                name='bn_')
            output_tensor = bn_layer(output_tensor, is_training, output_mask)

        if self.acti_func is not None:
            acti_layer = ActiLayer(
                func=self.acti_func,
                regularizer=self.regularizers['w'],
                name='acti_')
            output_tensor = acti_layer(output_tensor)
        return output_tensor, output_mask
