# -*- coding: utf-8 -*-
import tensorflow as tf
from base import Layer
from highresblock import HighResBlock
from convolution import ConvolutionalLayer
from six.moves import range
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
