# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from niftynet.layer.base_layer import Layer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_OP = set(['AVG', 'MAX'])

class SELayer(Layer):
    def __init__(self,func='AVG',reduction_ratio=16,name='squeeze_excitation'):
        self.func = func.upper()
        self.reduction_ratio=reduction_ratio
        self.layer_name = '{}_{}'.format(self.func.lower(), name)
        super(SELayer, self).__init__(name=self.layer_name)

        look_up_operations(self.func, SUPPORTED_OP)
        
    def layer_op(self, input_tensor):
        #squeeze: global information embedding
        input_tensor_shape=tf.shape(input_tensor).shape
        if input_tensor_shape==4:
            if self.func=='AVG':
                squeeze_tensor = tf.reduce_mean(input_tensor,axis=[1,2])
            elif self.func=='MAX':
                squeeze_tensor = tf.reduce_max(input_tensor,axis=[1,2])
        elif input_tensor_shape==5:
            if self.func=='AVG':
                squeeze_tensor = tf.reduce_mean(input_tensor,axis=[1,2,3])
            elif self.func=='MAX':
                squeeze_tensor = tf.reduce_max(input_tensor,axis=[1,2,3])
        else:
            raise ValueError("input shape not supported")
            
        #excitation: adaptive recalibration
        num_channels=int(squeeze_tensor.get_shape()[-1])
        reduction_ratio=self.reduction_ratio
        if num_channels % reduction_ratio != 0:
            raise ValueError("reduction ratio incompatible with number of input tensor channels")
        
        num_channels_reduced=num_channels/reduction_ratio
        fc1=FullyConnectedLayer(num_channels_reduced,with_bias=False,with_bn=False,acti_func='relu')
        fc2=FullyConnectedLayer(num_channels,with_bias=False,with_bn=False,acti_func='sigmoid')
        
        fc_out_1=fc1(squeeze_tensor)
        fc_out_2=fc2(fc_out_1)
        
        if input_tensor_shape==4:
            fc_out_2=fc_out_2[:,tf.newaxis,tf.newaxis,:]
        elif input_tensor_shape==5:
            fc_out_2=fc_out_2[:,tf.newaxis,tf.newaxis,tf.newaxis,:]
        
        output_tensor=tf.multiply(input_tensor,fc_out_2)
              
        return output_tensor
        
        
        
