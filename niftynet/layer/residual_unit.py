# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.activation import ActiLayer as Acti
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_OP = set(['original', 'conv_bn_acti', 'acti_conv_bn', 'bn_acti_conv'])


class ResidualUnit(TrainableLayer):
    def __init__(self,
                 n_output_chns=1,
                 kernel_size=3,
                 dilation=1,
                 acti_func='relu',
                 w_initializer=None,
                 w_regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 type_string='bn_acti_conv',
                 name='res-downsample'):
        """
        Implementation of residual unit presented in:

            [1] He et al., Identity mapping in deep residual networks, ECCV 2016
            [2] He et al., Deep residual learning for image recognition, CVPR 2016

        The possible types of connections are::

            'original': residual unit presented in [2]
            'conv_bn_acti': ReLU before addition presented in [1]
            'acti_conv_bn': ReLU-only pre-activation presented in [1]
            'bn_acti_conv': full pre-activation presented in [1]

        [1] recommends 'bn_acti_conv'

        :param n_output_chns: number of output feature channels
            if this doesn't match the input, a 1x1 projection will be created.
        :param kernel_size:
        :param dilation:
        :param acti_func:
        :param w_initializer:
        :param w_regularizer:
        :param moving_decay:
        :param eps:
        :param type_string:
        :param name:
        """

        super(TrainableLayer, self).__init__(name=name)
        self.type_string = look_up_operations(type_string.lower(), SUPPORTED_OP)
        self.acti_func = acti_func
        self.conv_param = {'w_initializer': w_initializer,
                           'w_regularizer': w_regularizer,
                           'kernel_size': kernel_size,
                           'dilation': dilation,
                           'n_output_chns': n_output_chns}
        self.bn_param = {'regularizer': w_regularizer,
                         'moving_decay': moving_decay,
                         'eps': eps}

    def layer_op(self, inputs, is_training=True):
        """
        The general connections is::

            (inputs)--o-conv_0--conv_1-+-- (outputs)
                      |                |
                      o----------------o

        ``conv_0``, ``conv_1`` layers are specified by ``type_string``.
        """
        conv_flow = inputs
        # batch normalisation layers
        bn_0 = BNLayer(**self.bn_param)
        bn_1 = BNLayer(**self.bn_param)
        # activation functions //regularisers?
        acti_0 = Acti(func=self.acti_func)
        acti_1 = Acti(func=self.acti_func)
        # convolutions
        conv_0 = Conv(acti_func=None, with_bias=False, feature_normalization=None,
                      **self.conv_param)
        conv_1 = Conv(acti_func=None, with_bias=False, feature_normalization=None,
                      **self.conv_param)

        if self.type_string == 'original':
            conv_flow = acti_0(bn_0(conv_0(conv_flow), is_training))
            conv_flow = bn_1(conv_1(conv_flow), is_training)
            conv_flow = ElementwiseLayer('SUM')(conv_flow, inputs)
            conv_flow = acti_1(conv_flow)
            return conv_flow

        if self.type_string == 'conv_bn_acti':
            conv_flow = acti_0(bn_0(conv_0(conv_flow), is_training))
            conv_flow = acti_1(bn_1(conv_1(conv_flow), is_training))
            return ElementwiseLayer('SUM')(conv_flow, inputs)

        if self.type_string == 'acti_conv_bn':
            conv_flow = bn_0(conv_0(acti_0(conv_flow)), is_training)
            conv_flow = bn_1(conv_1(acti_1(conv_flow)), is_training)
            return ElementwiseLayer('SUM')(conv_flow, inputs)

        if self.type_string == 'bn_acti_conv':
            conv_flow = conv_0(acti_0(bn_0(conv_flow, is_training)))
            conv_flow = conv_1(acti_1(bn_1(conv_flow, is_training)))
            return ElementwiseLayer('SUM')(conv_flow, inputs)

        raise ValueError('Unknown type string')
