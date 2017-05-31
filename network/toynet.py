# -*- coding: utf-8 -*-
from layer.base_layer import TrainableLayer
from layer.convolution import ConvolutionalLayer


class ToyNet(TrainableLayer):
    def __init__(self, num_classes, name='ToyNet'):
        super(ToyNet, self).__init__(name=name)
        self.hidden_features = 10
        self.num_classes = num_classes

    def layer_op(self, images, is_training):
        conv_1 = ConvolutionalLayer(self.hidden_features,
                                    kernel_size=3,
                                    acti_func='relu',
                                    name='conv_input')
        conv_2 = ConvolutionalLayer(self.num_classes,
                                    kernel_size=1,
                                    acti_func=None,
                                    name='conv_output')

        flow = conv_1(images, is_training)
        flow = conv_2(flow, is_training)
        return flow
