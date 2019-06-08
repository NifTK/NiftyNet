# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.network.base_net import BaseNet


class ToyNetFeat(BaseNet):
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='ToyNet'):

        super(ToyNetFeat, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.hidden_features = 10

    def layer_op(self, images, is_training=True, **unused_kwargs):
        conv_1 = ConvolutionalLayer(self.hidden_features,
                                    kernel_size=3,
                                    w_initializer=self.initializers['w'],
                                    w_regularizer=self.regularizers['w'],
                                    b_initializer=self.initializers['b'],
                                    b_regularizer=self.regularizers['b'],
                                    acti_func='relu',
                                    name='conv_input')

        # conv_2 = ConvolutionalLayer(self.num_classes,
        #                             kernel_size=1,
        #                             w_initializer=self.initializers['w'],
        #                             w_regularizer=self.regularizers['w'],
        #                             b_initializer=self.initializers['b'],
        #                             b_regularizer=self.regularizers['b'],
        #                             acti_func=None,
        #                             name='conv_output')

        flow = conv_1(images, is_training)
        # flow = conv_2(flow, is_training)
        return flow
