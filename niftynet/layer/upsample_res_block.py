# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.activation import ActiLayer as Acti
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.deconvolution import DeconvolutionalLayer as Deconv


class UpBlock(TrainableLayer):
    def __init__(self,
                 n_output_chns=4,
                 kernel_size=3,
                 upsample_stride=2,
                 acti_func='relu',
                 w_initializer=None,
                 w_regularizer=None,
                 name='res-upsample'):
        super(TrainableLayer, self).__init__(name=name)
        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.acti_func = acti_func
        self.upsample_stride = upsample_stride
        self.conv_param = {'w_initializer': w_initializer,
                           'w_regularizer': w_regularizer}

    def layer_op(self, inputs, forwarding=None, is_training=True):
        """
        inputs--deconv_0---+-+-conv_1--conv_2-+--
                           | |                |
        forwarding---------o o----------------o
        """
        deconv_0 = Deconv(n_output_chns=self.n_output_chns,
                          kernel_size=self.kernel_size,
                          stride=self.upsample_stride,
                          acti_func=self.acti_func,
                          with_bias=False, with_bn=True,
                          **self.conv_param)(inputs, is_training)
        conv_0 = deconv_0 if forwarding is None else deconv_0 + forwarding
        conv_1 = Conv(n_output_chns=self.n_output_chns,
                      kernel_size=self.kernel_size,
                      acti_func=self.acti_func,
                      with_bias=False, with_bn=True,
                      **self.conv_param)(conv_0, is_training)
        conv_2 = Conv(n_output_chns=self.n_output_chns,
                      kernel_size=self.kernel_size,
                      acti_func=None,
                      with_bias=False, with_bn=True,
                      **self.conv_param)(conv_1, is_training)
        conv_res = Acti(self.acti_func)(conv_2 + conv_0)
        return conv_res
