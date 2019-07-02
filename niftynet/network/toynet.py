# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.network.base_net import BaseNet


class ToyNet(BaseNet):
    """
    ### Description
        Toy net for testing

    ### Diagram
    INPUT --> CONV(kernel = 3, activation = relu) --> CONV(kernel = 1, activation = None) --> MULTICLASS OUTPUT

    ### Constraints
    None
    """
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='ToyNet'):
        """

        :param num_classes: int, number of final output channels
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param acti_func: ctivation function to use
        :param name: layer name
        """

        super(ToyNet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.hidden_features = 10

    def layer_op(self, images, is_training=True, **unused_kwargs):
        """

        :param images: tensor, input to the network
        :param is_training: boolean, True if network is in training mode
        :param unused_kwargs: other arguments, not in use
        :return: tensor, network output
        """
        conv_1 = ConvolutionalLayer(self.hidden_features,
                                    kernel_size=3,
                                    w_initializer=self.initializers['w'],
                                    w_regularizer=self.regularizers['w'],
                                    b_initializer=self.initializers['b'],
                                    b_regularizer=self.regularizers['b'],
                                    acti_func='relu',
                                    name='conv_input')

        conv_2 = ConvolutionalLayer(self.num_classes,
                                    kernel_size=1,
                                    w_initializer=self.initializers['w'],
                                    w_regularizer=self.regularizers['w'],
                                    b_initializer=self.initializers['b'],
                                    b_regularizer=self.regularizers['b'],
                                    acti_func=None,
                                    name='conv_output')

        flow = conv_1(images, is_training)
        flow = conv_2(flow, is_training)
        return flow
