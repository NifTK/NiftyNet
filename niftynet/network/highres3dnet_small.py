# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

from niftynet.layer import layer_util
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.network.base_net import BaseNet
from niftynet.network.highres3dnet import HighResBlock


class HighRes3DNetSmall(BaseNet):
    """
    implementation of HighRes3DNet:

        Li et al., "On the compactness, efficiency, and representation of 3D
        convolutional networks: Brain parcellation as a pretext task", IPMI '17

    (This is smaller model with an initial stride-2 convolution)

    ### Constraints
    - Input image size should be divisible by 8
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='HighRes3DNetSmall'):
        """

        :param num_classes: int, number of channels of output
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param acti_func: activation function to use
        :param name: layer name
        """

        super(HighRes3DNetSmall, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.layers = [
            {'name': 'conv_0', 'n_features': 16, 'kernel_size': 3},
            {'name': 'res_1', 'n_features': 16, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'res_2', 'n_features': 32, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'res_3', 'n_features': 64, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'conv_1', 'n_features': 80, 'kernel_size': 3},
            {'name': 'conv_2', 'n_features': num_classes, 'kernel_size': 1}]

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        """

        :param images: tensor to input to the network. Size has to be divisible by 8
        :param is_training: boolean, True if network is in training mode
        :param layer_id: int, index of the layer to return as output
        :param unused_kwargs:
        :return: output of layer indicated by layer_id
        """
        assert (layer_util.check_spatial_dims(
            images, lambda x: x % 8 == 0))
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []

        ### first convolution layer
        params = self.layers[0]
        first_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            stride=2,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = first_conv_layer(images, is_training)
        layer_instances.append((first_conv_layer, flow))

        ### resblocks, all kernels dilated by 1 (normal convolution)
        params = self.layers[1]
        with DilatedTensor(flow, dilation_factor=1) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### resblocks, all kernels dilated by 2
        params = self.layers[2]
        with DilatedTensor(flow, dilation_factor=2) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### resblocks, all kernels dilated by 4
        params = self.layers[3]
        with DilatedTensor(flow, dilation_factor=4) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### 1x1x1 convolution layer
        params = self.layers[4]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow))

        ### 3x3x3 deconvolution layer
        params = self.layers[4]
        fc_layer = DeconvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=3,
            stride=2,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='deconv')
        flow = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow))

        ### 1x1x1 convolution layer
        params = self.layers[5]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow))

        # set training properties
        if is_training:
            self._print(layer_instances)
            return layer_instances[-1][1]
        return layer_instances[layer_id][1]

    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)
