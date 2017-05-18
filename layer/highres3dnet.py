# -*- coding: utf-8 -*-
from six.moves import range

from . import layer_util
from .activation import ActiLayer
from .base import Layer
from .bn import BNLayer
from .convolution import ConvLayer, ConvolutionalLayer
from .dilatedcontext import DilatedTensor
from .elementwise import ElementwiseLayer


class HighRes3DNet(Layer):
    """
    implementation of HighRes3DNet:
      Li et al., "On the compactness, efficiency, and representation of 3D
      convolutional networks: Brain parcellation as a pretext task", IPMI '17
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 b_initializer=None,
                 w_regularizer=None,
                 b_regularizer=None,
                 acti_type='prelu',
                 name='HighRes3DNet'):

        super(HighRes3DNet, self).__init__(name='HighRes3DNet')
        self.layers = [
            {'name': 'conv_0', 'n_features': 16, 'kernel_size': 3},
            {'name': 'res_1', 'n_features': 16, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'res_2', 'n_features': 32, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'res_3', 'n_features': 64, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'conv_1', 'n_features': 80, 'kernel_size': 1},
            {'name': 'conv_2', 'n_features': num_classes, 'kernel_size': 1}]
        self.acti_type = acti_type
        self.w_initializer = w_initializer
        self.w_regularizer = w_regularizer
        self.b_initializer = b_initializer
        self.b_regularizer = b_regularizer
        self.name = "HighRes3DNet"
        print 'using {}'.format(self.name)

    def layer_op(self, images, is_training, layer_id=-1):
        assert (layer_util.check_spatial_dims(
            images, lambda x: x % 4 == 0))
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []

        ### first convolution layer
        params = self.layers[0]
        first_conv_layer = ConvolutionalLayer(params['n_features'],
                                              params['kernel_size'],
                                              name=params['name'])
        flow = first_conv_layer(images, is_training)
        layer_instances.append((first_conv_layer, flow))

        ### resblocks, all kernels dilated by 1 (normal convolution)
        params = self.layers[1]
        with DilatedTensor(flow, dilation_factor=1) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(params['n_features'],
                                         params['kernels'],
                                         name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### resblocks, all kernels dilated by 2
        params = self.layers[2]
        with DilatedTensor(flow, dilation_factor=2) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(params['n_features'],
                                         params['kernels'],
                                         name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### resblocks, all kernels dilated by 4
        params = self.layers[3]
        with DilatedTensor(flow, dilation_factor=4) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(params['n_features'],
                                         params['kernels'],
                                         name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### 1x1x1 convolution layer
        params = self.layers[4]
        fc_layer = ConvolutionalLayer(params['n_features'],
                                      params['kernel_size'],
                                      name=params['name'])
        flow = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow))

        ### 1x1x1 convolution layer
        params = self.layers[5]
        fc_layer = ConvolutionalLayer(params['n_features'],
                                      params['kernel_size'],
                                      name=params['name'])
        flow = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow))

        # set training properties
        if is_training:
            self._assign_initializer_regularizer(layer_instances)
            return layer_instances[-1][1]
        return layer_instances[layer_id][1]

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
        if hasattr(kernels, "__iter__"):  # a list of layer kernel_sizes
            self.kernels = kernels
        else:  # is a single number (indicating single layer)
            self.kernels = [kernels]
        self.acti_type = acti_type
        self.w_initializer = None
        self.w_regularizer = None
        self.layer_name = name
        self.with_res = with_res
        super(HighResBlock, self).__init__(name=self.layer_name)

    def layer_op(self, input_tensor, is_training):
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
