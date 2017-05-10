# -*- coding: utf-8 -*-
from six.moves import range

import tensorflow as tf
import numpy as np
from bn import BNLayer
from base import Layer
from convolution import ConvLayer, ConvolutionalLayer
from activation import ActiLayer
from elementwise import ElementwiseLayer
from dilatedcontext import DilatedTensor


"""
implementation of HighRes3DNet:
  Li et al., "On the compactness, efficiency, and representation of 3D
  convolutional networks: Brain parcellation as a pretext task", IPMI '17
"""
class HighRes3DNet(Layer):
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 b_initializer=None,
                 w_regularizer=None,
                 b_regularizer=None,
                 acti_type='prelu',
                 name='HighRes3DNet'):

        super(HighRes3DNet, self).__init__(name='HighRes3DNet')
        self.model = [
                {'op': 'conv', 'n_features': 16, 'kernel_size': 3},
                {'op': 'resblocks', 'dilation_factor': 1, 'repeat': 3, 'n_features': 16, 'kernels':(3, 3)},
                {'op': 'resblocks', 'dilation_factor': 2, 'repeat': 3, 'n_features': 32, 'kernels':(3, 3)},
                {'op': 'resblocks', 'dilation_factor': 4, 'repeat': 3, 'n_features': 64, 'kernels':(3, 3)},
                {'op': 'conv', 'n_features': 80, 'kernel_size': 1},
                {'op': 'conv', 'n_features': num_classes, 'kernel_size': 1}]
        self.acti_type = acti_type
        self.w_initializer = w_initializer
        self.w_regularizer = w_regularizer
        self.b_initializer = b_initializer
        self.b_regularizer = b_regularizer
        self.name = "HighRes3DNet"
        print('using {}'.format(self.name))


    def layer_op(self, images, is_training, layer_id=-1):
        assert(images.get_shape()[1] % 4 == 0)
        # create operations
        flow = images
        list_of_layers = []
        for (i, layer) in enumerate(self.model):
            # create convolution layers
            if layer['op'] == 'conv':
                # creat a convolutional layer
                op_ = ConvolutionalLayer(layer['n_features'],
                                         kernel_size=layer['kernel_size'],
                                         name='conv_{}'.format(i))
                # data running through the layer
                flow = op_(flow, is_training)
                list_of_layers.append((op_, flow))
            # create resblocks
            if layer['op'] == 'resblocks':
                # instead of dilating the kernels, the dilated convolution
                # is implmeneted by rearranging the input tensor
                # The arrangment of input is resumed after the context manager
                with DilatedTensor(flow, layer['dilation_factor']) as dilated:
                    for j in range(layer['repeat']):
                        # creat a highresblock layer
                        op_ = HighResBlock(layer['n_features'],
                                           kernels=layer['kernels'],
                                           name='res_{}_{}'.format(i, j))
                        # data running through the layer
                        dilated.tensor = op_(dilated.tensor, is_training)
                        list_of_layers.append((op_, dilated.tensor))
                # resume to the ordinary flow after the context manager
                flow = dilated.tensor
            if (i == layer_id) and (not is_training):
                return flow

        if is_training:
            self._assign_initializer_regularizer(list_of_layers)
        return flow

    def _assign_initializer_regularizer(self, list_of_layers):
        for (op, _) in list_of_layers:
            print op
            op.w_initializer = self.w_initializer
            op.w_regularizer = self.w_regularizer
            op.b_initializer = self.b_initializer
            op.b_regularizer = self.b_regularizer



class HighResBlock(Layer):
    """
    This class define a high-resolution block with residual connections
    kernels - specify kernel sizes of each convolutional layer
            - e.g.: kernels=(5, 5, 5) indicate three conv layers of kernel_size 5
    with_res - whether to add residual connections to bypass the conv layers
    """
    def __init__(self,
                 n_output_chns,
                 kernels=(3, 3),
                 acti_type='relu',
                 w_initializer=None,
                 w_regularizer=None,
                 with_res=True,
                 name='HighResBlock'):
        self.n_output_chns = n_output_chns
        if hasattr(kernels, "__iter__"):  # is a list of layer kernel_sizes
            self.kernels = kernels
        else:  # is a single number (indicating single layer)
            self.kernels = [kernels]
        self.acti_type = acti_type
        self.w_initializer=None
        self.w_regularizer=None
        self.layer_name = name
        self.with_res = with_res
        super(HighResBlock, self).__init__(name=self.layer_name)

    def layer_op(self, input_tensor, is_training):
        self.n_input_chns = input_tensor.get_shape()[-1]
        output_tensor = input_tensor
        for (i, k) in enumerate(self.kernels):
            # create parameterised layers
            bn_op = BNLayer(regularizer=self.w_regularizer,
                            name='bn_{}'.format(i))
            acti_op = ActiLayer(func=self.acti_type,
                                regularizer=self.w_regularizer,
                                name='acti_{}'.format(i))
            conv_op = ConvLayer(n_output_chns=self.n_output_chns,
                                kernel_size=k,
                                stride=1,
                                w_initializer=self.w_initializer,
                                w_regularizer=self.w_regularizer,
                                name='conv_{}'.format(i))

            # connect layers
            output_tensor = bn_op(output_tensor, is_training)
            output_tensor = acti_op(output_tensor)
            output_tensor = conv_op(output_tensor)

        # make residual connections
        if self.with_res:
            output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)
        return output_tensor
