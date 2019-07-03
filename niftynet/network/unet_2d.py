# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer, Layer
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.crop import CropLayer as Crop
from niftynet.layer.deconvolution import DeconvolutionalLayer as DeConv
from niftynet.layer.downsample import DownSampleLayer as Pooling
from niftynet.layer.elementwise import ElementwiseLayer as ElementWise
from niftynet.layer.linear_resize import LinearResizeLayer as Resize
from niftynet.network.base_net import BaseNet


class UNet2D(BaseNet):
    """
    ### Description
        A reimplementation of 2D UNet:
            Ronneberger et al., U-Net: Convolutional Networks for Biomedical
            Image Segmentation, MICCAI '15

    ### Building blocks
    [dBLOCK]        - Downsampling Block (conv 3x3, Relu + conv 3x3, Relu +  Max pooling)
    [BLOCK]         - Two layer Block (conv 3x3, Relu + conv 3x3, Relu)
    [uBLOCK]        - Upsampling Block (deconv 2x2 + crop and concat + conv 3x3, Relu + conv 3x3, Relu)
    [CONV]          - Classification block (conv 1x1)

    ### Diagram

    INPUT  -->  [dBLOCK] - - - - - - - - - - - - - - - -  [BLOCK] --> [CONV] --> OUTPUT
                    |                                       |
                    [dBLOCK] - - - - - - - - - - - - [uBLOCK]
                        |                              |
                        [dBLOCK]  - - - - - - - [uBLOCK]
                            |                      |
                            [dBLOCK]  - - - [uBLOCK]
                                |               |
                                ----[BLOCk] ----

    ### Constraints
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
                         name=name)
        self.n_fea = [64, 128, 256, 512, 1024]

        net_params = {'padding': 'VALID',
                      'with_bias': True,
                      'feature_normalization': 'batch',
                      'group_size': -1,
                      'acti_func': acti_func,
                      'w_initializer': w_initializer,
                      'b_initializer': b_initializer,
                      'w_regularizer': w_regularizer,
                      'b_regularizer': b_regularizer}

        self.conv_params = {'kernel_size': 3, 'stride': 1}
        self.deconv_params = {'kernel_size': 2, 'stride': 2}
        self.pooling_params = {'kernel_size': 2, 'stride': 2}

        self.conv_params.update(net_params)
        self.deconv_params.update(net_params)

    def layer_op(self, images, is_training=True, **unused_kwargs):
        """

        :param images: tensor, input to the network
        :param is_training: boolean, True if network is in training mode
        :param unused_kwargs: other conditional arguments, not in use
        :return: tensor, output of the network
        """
        # contracting path
        output_1 = TwoLayerConv(self.n_fea[0], self.conv_params)(images, is_training=is_training)
        down_1 = Pooling(func='MAX', **self.pooling_params)(output_1)

        output_2 = TwoLayerConv(self.n_fea[1], self.conv_params)(down_1, is_training=is_training)
        down_2 = Pooling(func='MAX', **self.pooling_params)(output_2)

        output_3 = TwoLayerConv(self.n_fea[2], self.conv_params)(down_2, is_training=is_training)
        down_3 = Pooling(func='MAX', **self.pooling_params)(output_3)

        output_4 = TwoLayerConv(self.n_fea[3], self.conv_params)(down_3, is_training=is_training)
        down_4 = Pooling(func='MAX', **self.pooling_params)(output_4)

        output_5 = TwoLayerConv(self.n_fea[4], self.conv_params)(down_4, is_training=is_training)

        # expansive path
        up_4 = DeConv(self.n_fea[3], **self.deconv_params)(output_5, is_training=is_training)
        output_4 = CropConcat()(output_4, up_4)
        output_4 = TwoLayerConv(self.n_fea[3], self.conv_params)(output_4, is_training=is_training)

        up_3 = DeConv(self.n_fea[2], **self.deconv_params)(output_4, is_training=is_training)
        output_3 = CropConcat()(output_3, up_3)
        output_3 = TwoLayerConv(self.n_fea[2], self.conv_params)(output_3, is_training=is_training)

        up_2 = DeConv(self.n_fea[1], **self.deconv_params)(output_3, is_training=is_training)
        output_2 = CropConcat()(output_2, up_2)
        output_2 = TwoLayerConv(self.n_fea[1], self.conv_params)(output_2, is_training=is_training)

        up_1 = DeConv(self.n_fea[0], **self.deconv_params)(output_2, is_training=is_training)
        output_1 = CropConcat()(output_1, up_1)
        output_1 = TwoLayerConv(self.n_fea[0], self.conv_params)(output_1, is_training=is_training)

        # classification layer
        classifier = Conv(n_output_chns=self.num_classes,
                          kernel_size=1,
                          with_bias=True,
                          feature_normalization=None)
        output_tensor = classifier(output_1)
        tf.logging.info('output shape %s', output_tensor.shape)
        return output_tensor


class TwoLayerConv(TrainableLayer):
    """
    Two convolutional layers, number of output channels are ``n_chns`` for both
    of them.

    --conv--conv--
    """

    def __init__(self, n_chns, conv_params):
        TrainableLayer.__init__(self, name='TwoConv')
        self.n_chns = n_chns
        self.conv_params = conv_params

    def layer_op(self, input_tensor, is_training=None):
        """

        :param input_tensor: tensor, input to the two layer convolution
        :param is_training: flag for training
        :return: tensor, output of --conv--conv
        """
        output_tensor = Conv(self.n_chns, **self.conv_params)(input_tensor, is_training=is_training)
        output_tensor = Conv(self.n_chns, **self.conv_params)(output_tensor, is_training=is_training)

        return output_tensor


class CropConcat(Layer):
    """
    This layer concatenates two input tensors,
    the first one is cropped and resized to match the second one.

    This layer assumes the same amount of differences
    in every spatial dimension in between the two tensors.
    """

    def __init__(self, name='crop_concat'):
        Layer.__init__(self, name=name)

    def layer_op(self, tensor_a, tensor_b):
        """
        match the spatial shape and concatenate the tensors
        tensor_a will be cropped and resized to match tensor_b.

        :param tensor_a: tensor, input
        :param tensor_b: tensor, input
        :return: concatenated tensor
        """
        crop_border = (tensor_a.shape[1] - tensor_b.shape[1]) // 2
        tensor_a = Crop(border=crop_border)(tensor_a)
        output_spatial_shape = tensor_b.shape[1:-1]
        tensor_a = Resize(new_size=output_spatial_shape)(tensor_a)
        return ElementWise('CONCAT')(tensor_a, tensor_b)
