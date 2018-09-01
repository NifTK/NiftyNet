# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import functools
from collections import namedtuple

import tensorflow as tf

from niftynet.layer.bn import BNLayer
from niftynet.layer.fully_connected import FCLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.squeeze_excitation import ChannelSELayer
from niftynet.network.base_net import BaseNet

SE_ResNetDesc = namedtuple('SE_ResNetDesc', ['bn', 'fc', 'conv1', 'blocks'])
class SE_ResNet(BaseNet):
    """
    3D implementation of SE-ResNet:
      Hu et al., "Squeeze-and-Excitation Networks", arXiv:1709.01507v2
    """

    def __init__(self,
                 num_classes,
                 n_features = [16, 64, 128],
                 n_blocks_per_resolution = 1,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='SE_ResNet'):

        super(SE_ResNet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.n_features = n_features
        self.n_blocks_per_resolution = n_blocks_per_resolution
        self.Conv = functools.partial(ConvolutionalLayer,
                                      w_initializer=w_initializer,
                                      w_regularizer=w_regularizer,
                                      b_initializer=b_initializer,
                                      b_regularizer=b_regularizer,
                                      preactivation=True,
                                      acti_func=acti_func)

    def create(self):
        bn=BNLayer()
        fc=FCLayer(self.num_classes)
        conv1=self.Conv(self.n_features[0], acti_func=None, with_bn=False)
        blocks=[]
        blocks+=[DownResBlock(self.n_features[1], self.n_blocks_per_resolution, 1, self.Conv)]
        for n in self.n_features[2:]:
            blocks+=[DownResBlock(n, self.n_blocks_per_resolution, 2, self.Conv)]
        return SE_ResNetDesc(bn=bn,fc=fc,conv1=conv1,blocks=blocks)

    def layer_op(self, images, is_training=True, **unused_kwargs):
        layers = self.create()
        out = layers.conv1(images, is_training)
        for block in layers.blocks:
            out = block(out, is_training)
        out = tf.reduce_mean(tf.nn.relu(layers.bn(out, is_training)),axis=[1,2,3])
        return layers.fc(out)
        
BottleneckBlockDesc1 = namedtuple('BottleneckBlockDesc1', ['conv'])
BottleneckBlockDesc2 = namedtuple('BottleneckBlockDesc2', ['common_bn', 'conv', 'conv_shortcut'])
class BottleneckBlock(TrainableLayer):
    def __init__(self, n_output_chns, stride, Conv, name='bottleneck'):
        self.n_output_chns = n_output_chns
        self.stride=stride
        self.bottle_neck_chns = n_output_chns // 4
        self.Conv = Conv
        super(BottleneckBlock, self).__init__(name=name)
        
    def create(self, input_chns):
        if self.n_output_chns == input_chns:
            b1 = self.Conv(self.bottle_neck_chns, kernel_size=1,
                           stride=self.stride)
            b2 = self.Conv(self.bottle_neck_chns, kernel_size=3)
            b3 = self.Conv(self.n_output_chns, 1)
            return BottleneckBlockDesc1(conv=[b1, b2, b3])
        else:
            b1 = BNLayer()
            b2 = self.Conv(self.bottle_neck_chns,kernel_size=1,
                           stride=self.stride, acti_func=None, with_bn=False)
            b3 = self.Conv(self.bottle_neck_chns,kernel_size=3)
            b4 = self.Conv(self.n_output_chns,kernel_size=1)
            b5 = self.Conv(self.n_output_chns,kernel_size=1,
                           stride=self.stride, acti_func=None,with_bn=False)
            return BottleneckBlockDesc2(common_bn=b1, conv=[b2, b3, b4], 
                              conv_shortcut=b5)

    def layer_op(self, images, is_training=True):
        layers = self.create(images.shape[-1])
        se=ChannelSELayer()
        if self.n_output_chns == images.shape[-1]:
            out=layers.conv[0](images, is_training)
            out=layers.conv[1](out, is_training)
            out=layers.conv[2](out, is_training)
            out=se(out)
            out = out+images
        else:
            tmp = tf.nn.relu(layers.common_bn(images, is_training))
            out=layers.conv[0](tmp, is_training)
            out=layers.conv[1](out, is_training)
            out=layers.conv[2](out, is_training)
            out=se(out)
            out = layers.conv_shortcut(tmp, is_training) + out
        print(out.shape)
        return out

DownResBlockDesc = namedtuple('DownResBlockDesc', ['blocks'])
class DownResBlock(TrainableLayer):
    def __init__(self, n_output_chns, count, stride, Conv, name='downres'):
        self.count = count
        self.stride = stride
        self.n_output_chns = n_output_chns
        self.Conv=Conv
        super(DownResBlock, self).__init__(name=name)
        
    def create(self):
        blocks=[]
        blocks+=[BottleneckBlock(self.n_output_chns, self.stride, self.Conv)]
        for it in range(1,self.count):
            blocks+=[BottleneckBlock(self.n_output_chns, 1, self.Conv)]
        return DownResBlockDesc(blocks=blocks)
        
    def layer_op(self, images, is_training):
        layers = self.create()
        out = images
        for l in layers.blocks:
            out=l(out,is_training)
        return out
