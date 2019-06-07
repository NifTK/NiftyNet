# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.additive_upsample import ResidualUpsampleLayer as ResUp
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer as Deconv
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.residual_unit import ResidualUnit as ResUnit


class UpBlock(TrainableLayer):
    def __init__(self,
                 n_output_chns=4,
                 kernel_size=3,
                 upsample_stride=2,
                 acti_func='relu',
                 w_initializer=None,
                 w_regularizer=None,
                 is_residual_upsampling=True,
                 type_string='bn_acti_conv',
                 name='res-upsample'):
        super(TrainableLayer, self).__init__(name=name)
        self.type_string = type_string
        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.acti_func = acti_func
        self.upsample_stride = upsample_stride
        self.conv_param = {'w_initializer': w_initializer,
                           'w_regularizer': w_regularizer}
        self.is_residual_upsampling = is_residual_upsampling

    def layer_op(self, inputs, forwarding=None, is_training=True):
        """
        Consists of::

            (inputs)--upsampling-+-o--conv_1--conv_2--+--(conv_res)--
                                 | |                  |
            (forwarding)---------o o------------------o

        where upsampling method could be ``DeconvolutionalLayer``
        or ``ResidualUpsampleLayer``
        """
        if self.is_residual_upsampling:
            n_input_channels = inputs.get_shape().as_list()[-1]
            n_splits = float(n_input_channels) / float(self.n_output_chns)
            upsampled = ResUp(kernel_size=self.kernel_size,
                              stride=self.upsample_stride,
                              n_splits=n_splits,
                              acti_func=self.acti_func,
                              **self.conv_param)(inputs, is_training)
        else:
            upsampled = Deconv(n_output_chns=self.n_output_chns,
                               kernel_size=self.kernel_size,
                               stride=self.upsample_stride,
                               acti_func=self.acti_func,
                               with_bias=False, feature_normalization='batch',
                               **self.conv_param)(inputs, is_training)

        if forwarding is None:
            conv_0 = upsampled
        else:
            conv_0 = ElementwiseLayer('SUM')(upsampled, forwarding)

        conv_res = ResUnit(n_output_chns=self.n_output_chns,
                           kernel_size=self.kernel_size,
                           acti_func=self.acti_func,
                           type_string=self.type_string,
                           **self.conv_param)(conv_0, is_training)
        return conv_res
