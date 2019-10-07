# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.layer_util import infer_spatial_rank
from niftynet.layer.fully_connected import FullyConnectedLayer
# from niftynet.layer.pool_full import PoolingLayer
import tensorflow as tf


class ClassSegFinnet(BaseNet):
    """
    implementation of HighRes3DNet:
      Li et al., "On the compactness, efficiency, and representation of 3D
      convolutional networks: Brain parcellation as a pretext task", IPMI '17
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='FinalClassSeg'):

        super(ClassSegFinnet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)
        self.num_classes = num_classes
        self.layers = [
            {'name': 'fc_seg', 'n_features': 10,
             'kernel_size': 1},
            {'name': 'pool', 'n_features': 10,
             'stride': 1, 'func': 'AVG'},
            {'name': 'fc_seg', 'n_features': num_classes,
             'kernel_size': 1},
            {'name': 'fc_class', 'n_features': 2, 'kernel_size': 1}]

    def layer_op(self, images, is_training, layer_id=-1):
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []

        # class convolution layer
        params = self.layers[0]
        fc_seg = ConvolutionalLayer(
            with_bn=False,
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=self.acti_func,
            padding='VALID',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fc_seg(images, is_training)
        layer_instances.append((fc_seg, flow))

        # pooling layer
        params = self.layers[1]
        # pool_layer = PoolingLayer(
        #     func=params['func'],
        #     name=params['name'])
        # flow_pool = pool_layer(flow)
        flow_pool = flow
        flow_pool = tf.reshape(flow_pool, [tf.shape(images)[0], 1, 1, 1,
                                           self.layers[1][
            'n_features']])
        print("check flow pooling", flow_pool.shape)
        layer_instances.append((pool_layer, flow_pool))

        # seg convolution layer
        params = self.layers[2]
        seg_conv_layer = ConvolutionalLayer(
            with_bn=False,
            n_output_chns=params['n_features'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            kernel_size=1,
            name=params['name'])
        seg_flow = seg_conv_layer(flow, is_training)
        layer_instances.append((seg_conv_layer, seg_flow))

        # class convolution layer
        params = self.layers[3]
        class_conv_layer = ConvolutionalLayer(
            with_bn=False,
            n_output_chns=params['n_features'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            kernel_size=1,
            name=params['name'])
        class_flow = class_conv_layer(flow_pool, is_training)
        layer_instances.append((class_conv_layer, class_flow))

        # set training properties
        if is_training:
            self._print(layer_instances)
            return layer_instances[-2][1], layer_instances[-1][1]
        return layer_instances[-2][1], layer_instances[layer_id][1]

    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)
