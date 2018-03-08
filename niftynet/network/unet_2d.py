# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.deconvolution import DeconvolutionalLayer as DeConv
from niftynet.layer.downsample import DownSampleLayer as Pooling
from niftynet.layer.crop import CropLayer as Crop
from niftynet.layer.elementwise import ElementwiseLayer as ElementWise
from niftynet.layer.linear_resize import LinearResizeLayer as Resize
from niftynet.network.base_net import BaseNet


class UNet2D(BaseNet):
    """
    A reimplementation of 2D UNet
    TODO: regulariser, initialiser
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='UNet2D'):
        BaseNet.__init__(self,
                         num_classes=num_classes,
                         w_initializer=w_initializer,
                         w_regularizer=w_regularizer,
                         b_initializer=b_initializer,
                         b_regularizer=b_regularizer,
                         acti_func=acti_func,
                         name=name)

        self.conv_params = {'kernel_size': 3,
                            'stride': 1,
                            'padding': 'VALID',
                            'with_bias': True,
                            'with_bn': False,
                            'acti_func': acti_func}

        self.deconv_params = {'kernel_size': 2,
                              'stride': 2,
                              'padding': 'VALID',
                              'with_bias': True,
                              'with_bn': False,
                              'acti_func': acti_func}

    def layer_op(self, images, is_training=None):

        # contracting path
        output_1 = TwoLayerConv(64, self.conv_params)(images)
        down_1 = Pooling(func='MAX', kernel_size=2)(output_1)

        output_2 = TwoLayerConv(128, self.conv_params)(down_1)
        down_2 = Pooling(func='MAX', kernel_size=2)(output_2)

        output_3 = TwoLayerConv(256, self.conv_params)(down_2)
        down_3 = Pooling(func='MAX', kernel_size=2)(output_3)

        output_4 = TwoLayerConv(512, self.conv_params)(down_3)
        down_4 = Pooling(func='MAX', kernel_size=2)(output_4)

        output_5 = TwoLayerConv(1024, self.conv_params)(down_4)

        # expansive path
        up_4 = DeConv(n_output_chns=512, **self.deconv_params)(output_5)
        border_4 = (output_4.shape[1] - up_4.shape[1]) // 2
        output_4 = ElementWise('CONCAT')(Crop(border=border_4)(output_4), up_4)
        output_4 = TwoLayerConv(512, self.conv_params)(output_4)

        up_3 = DeConv(n_output_chns=256, **self.deconv_params)(output_4)
        border_3 = (output_3.shape[1] - up_3.shape[1]) // 2
        output_3 = Resize(up_3.shape[1:-1])(Crop(border=border_3)(output_3))
        output_3 = ElementWise('CONCAT')(output_3, up_3)
        output_3 = TwoLayerConv(256, self.conv_params)(output_3)

        up_2 = DeConv(n_output_chns=128, **self.deconv_params)(output_3)
        border_2 = (output_2.shape[1] - up_2.shape[1]) // 2
        output_2 = Resize(up_2.shape[1:-1])(Crop(border=border_2)(output_2))
        output_2 = ElementWise('CONCAT')(output_2, up_2)
        output_2 = TwoLayerConv(128, self.conv_params)(output_2)

        up_1 = DeConv(n_output_chns=64, **self.deconv_params)(output_2)
        border_1 = (output_1.shape[1] - up_1.shape[1]) // 2
        output_1 = Resize(up_1.shape[1:-1])(Crop(border=border_1)(output_1))
        output_1 = ElementWise('CONCAT')(output_1, up_1)
        output_1 = TwoLayerConv(64, self.conv_params)(output_1)

        # classification layer
        classifier = Conv(n_output_chns=self.num_classes,
                          kernel_size=1, with_bias=True, with_bn=False)
        output_tensor = classifier(output_1)
        return output_tensor


class TwoLayerConv(TrainableLayer):
    """
    Two convolutional layers, number of output channels are ``n_chns`` for both
    of them.

    --conv--conv--
    """

    def __init__(self, n_chns, conv_params):
        TrainableLayer.__init__(self)
        self.n_chns = n_chns
        self.conv_params = conv_params

    def layer_op(self, input_tensor):
        output_tensor = Conv(self.n_chns, **self.conv_params)(input_tensor)
        output_tensor = Conv(self.n_chns, **self.conv_params)(output_tensor)
        return output_tensor
