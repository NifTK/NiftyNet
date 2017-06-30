# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import layer.convolution
import layer.deconvolution
import layer.bn
from tensorflow.python.training import moving_averages
from utilities.misc_common import look_up_operations
from . import layer_util
import numpy as np

class ChannelSparseDeconvLayer(layer.deconvolution.DeconvLayer):
  def __init__(self,*args,**kwargs):
    super(ChannelSparseDeconvLayer,self).__init__(*args,**kwargs)
  def layer_op(self,input_tensor,input_mask,output_mask):
    input_shape = input_tensor.get_shape().as_list()
    n_full_input_chns = input_mask.get_shape().as_list()[0]
    n_sparse_output_chns = tf.reduce_sum(tf.cast(output_mask, tf.float32))
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
    sparse_kernel = tf.transpose(tf.boolean_mask(
                       tf.transpose(tf.boolean_mask(
                         tf.transpose(deconv_kernel,[3,4,2,1,0]),output_mask),[1,0,2,3,4]),input_mask),[4,3,2,1,0])
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
    sparse_output_size = tf.stack([input_shape[0],
                                  [output_dim] * spatial_rank,
                                  n_sparse_output_chns],0)
    output_tensor = op_(value=input_tensor,
                        filter=deconv_kernel,
                        output_shape=sparse_output_size,
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
    sparse_bias = tf.boolean_mask(bias_term,output_mask)

    output_tensor = tf.nn.bias_add(output_tensor,
                                   sparse_bias,
                                   name='add_bias')
    return output_tensor

class ChannelSparseConvLayer(layer.convolution.ConvLayer):
  def __init__(self,*args,**kwargs):
    super(ChannelSparseConvLayer,self).__init__(*args,**kwargs)
  def layer_op(self,input_tensor,input_mask,output_mask):
    sparse_input_shape = input_tensor.get_shape().as_list()
    n_full_input_chns = input_mask.get_shape().as_list()[0]
    spatial_rank = layer_util.infer_spatial_rank(input_tensor)
    # initialize conv kernels/strides and then apply
    w_full_size = np.vstack((
        [self.kernel_size] * spatial_rank,
        n_full_input_chns, self.n_output_chns)).flatten()
    full_stride = np.vstack((
        [self.stride] * spatial_rank)).flatten()
    conv_kernel = tf.get_variable(
        'w', shape=w_full_size.tolist(),
        initializer=self.initializers['w'],
        regularizer=self.regularizers['w'])
    sparse_kernel = tf.transpose(tf.boolean_mask(
                       tf.transpose(tf.boolean_mask(
                         tf.transpose(conv_kernel,[4,3,2,1,0]),
                         output_mask),[1,0,2,3,4]),input_mask),[4,3,2,0,1])
    output_tensor = tf.nn.convolution(input=input_tensor,
                                      filter=sparse_kernel,
                                      strides=full_stride.tolist(),
                                      padding=self.padding,
                                      name='conv')
    if not self.with_bias:
        return output_tensor

    # adding the bias term
    bias_term = tf.get_variable(
        'b', shape=self.n_output_chns,
        initializer=self.initializers['b'],
        regularizer=self.regularizers['b'])
    sparse_bias = tf.boolean_mask(bias_term,output_mask)
    output_tensor = tf.nn.bias_add(output_tensor, bias_term,
                                       name='add_bias')
    return output_tensor

class ChannelSparseBNLayer(layer.bn.BNLayer):
  def __init__(self,*args,**kwargs):
    super(ChannelSparseBNLayer,self).__init__(*args,**kwargs)  
  def layer_op(self, inputs, is_training, mask, use_local_stats=False):
    input_shape = inputs.get_shape()
    mask_shape = mask.get_shape()

    # operates on all dims except the last dim
    params_shape = mask_shape[-1:]
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
    mean_update=tf.dynamic_stitch([tf.to_int32(tf.where(mask)[:,0]),tf.to_int32(tf.where(~mask)[:,0])],[mean,tf.boolean_mask(moving_mean,~mask)])
    variance_update=tf.dynamic_stitch([tf.to_int32(tf.where(mask)[:,0]),tf.to_int32(tf.where(~mask)[:,0])],[variance,tf.boolean_mask(moving_variance,~mask)])
    
    update_moving_mean = moving_averages.assign_moving_average(
         moving_mean, mean_update, self.moving_decay).op
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance_update, self.moving_decay).op
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

    # call the normalisation function
    if is_training or use_local_stats:
        # with tf.control_dependencies(
        #         [update_moving_mean, update_moving_variance]):
        outputs = tf.nn.batch_normalization(
            inputs, mean, variance,
            tf.boolean_mask(beta,mask), tf.boolean_mask(gamma,mask), self.eps, name='batch_norm')
    else:
        outputs = tf.nn.batch_normalization(
            inputs, tf.boolean_mask(moving_mean,mask), tf.boolean_mask(moving_variance,mask),
            tf.boolean_mask(beta,mask), tf.boolean_mask(gamma,mask), self.eps, name='batch_norm')
    outputs.set_shape(inputs.get_shape())
    return outputs

  