# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.deconvolution import DeconvolutionalLayer as DeConv
from niftynet.layer.downsample import DownSampleLayer as Down
from niftynet.layer.crop import CropLayer as Crop
from niftynet.layer.elementwise import ElementwiseLayer as ElementWise
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

        self.deconv_params = {'kernel_size': 2,
                              'stride': 1,
                              'padding': 'VALID',
                              'with_bias': True,
                              'with_bn': False}

    def layer_op(self, images, is_training=None):

        # contracting path
        output_L1 = TwoLayerConv(n_chns=64)(images)
        down_L1 = Down(func='MAX', kernel_size=2)(output_L1)

        output_L2 = TwoLayerConv(n_chns=128)(down_L1)
        down_L2 = Down(func='MAX', kernel_size=2)(output_L2)

        output_L3 = TwoLayerConv(n_chns=256)(down_L2)
        down_L3 = Down(func='MAX', kernel_size=2)(output_L3)

        output_L4 = TwoLayerConv(n_chns=512)(down_L3)
        down_L4 = Down(func='MAX', kernel_size=2)(output_L4)

        output_L5 = TwoLayerConv(n_chns=1024)(down_L4)

        # expansive path
        up_L4 = DeConv(n_output_chns=512, **self.deconv_params)(output_L5)
        output_L4 = ElementWise('CONCAT')(Crop(border=4)(output_L4), up_L4)
        output_L4 = TwoLayerConv(n_chns=512)(output_L4)

        up_L3 = DeConv(n_output_chns=256, **self.deconv_params)(output_L4)
        output_L3 = ElementWise('CONCAT')(Crop(border=4)(output_L3), up_L3)
        output_L3 = TwoLayerConv(n_chns=256)(output_L3)

        up_L2 = DeConv(n_output_chns=128, **self.deconv_params)(output_L3)
        output_L2 = ElementWise('CONCAT')(Crop(border=4)(output_L2), up_L2)
        output_L2 = TwoLayerConv(n_chns=128)(output_L2)

        up_L1 = Deconv(n_output_chns=64, **self.deconv_params)(output_L2)
        output_L1 = ElementWise('CONCAT')(Crop(border=4)(output_L1), up_L1)
        output_L1 = TwoLayerConv(n_chns=64)(output_L1)

        # classification layer
        Classifier = Conv(n_output_chns=self.num_classes,
                          kernel_size=1, with_bias=True, with_bn=False)
        return Classifier(output_L1)


class TwoLayerConv(TrainableLayer):

    def __init__(self, n_chns):
        TrainableLayer.__init__(self)
        self.conv_params = {
            'n_output_chns': n_chns,
            'kernel_size': 3,
            'stride': 1,
            'padding': 'VALID',
            'with_bias': True,
            'with_bn': False}


    def layer_op(self, input_tensor):
        output_tensor = Conv(**self.conv_params)(input_tensor)
        output_tensor = Conv(**self.conv_params)(output_tensor)
        return output_tensor
