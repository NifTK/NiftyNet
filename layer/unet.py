# -*- coding: utf-8 -*-
import tensorflow as tf

from base import Layer
from convolution import ConvolutionalLayer
from deconvolution import DeconvolutionalLayer
from downsample import DownSampleLayer
from upsample import UpSampleLayer
from elementwise import ElementwiseLayer
import layer_util


"""
reimplementation of 3D U-net
  Çiçek et al., "3D U-Net: Learning dense Volumetric segmentation from
  sparse annotation", MICCAI '16
"""
class UNet3D(Layer):
    def __init__(self, num_classes):
        self.layer_name = 'UNet_3D'
        super(UNet3D, self).__init__(name=self.layer_name)
        self.n_features = [32, 64, 128, 256, 512]
        self.acti_type = 'relu'
        self.num_classes = num_classes

    def layer_op(self, images, is_training, layer_id=-1):
        # image_size  should be divisible by 8
        assert(layer_util.check_spatial_dims(images, lambda x: x % 8 == 0))
        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[0], self.n_features[1]),
                                (3, 3), with_downsample_branch=True,
                                name='L1')
        pool_1, conv_1 = block_layer(images, is_training)
        print block_layer

        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[1], self.n_features[2]),
                                (3, 3), with_downsample_branch=True,
                                name='L2')
        pool_2, conv_2 = block_layer(pool_1, is_training)
        print block_layer

        block_layer = UNetBlock('DOWNSAMPLE',
                                (self.n_features[2], self.n_features[3]),
                                (3, 3), with_downsample_branch=True,
                                name='L3')
        pool_3, conv_3 = block_layer(pool_2, is_training)
        print block_layer

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[3], self.n_features[4]),
                                (3, 3), with_downsample_branch=False,
                                name='L4')
        up_3, _ = block_layer(pool_3, is_training)
        print block_layer

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[3], self.n_features[3]),
                                (3, 3), with_downsample_branch=False,
                                name='R3')
        concat_3 = ElementwiseLayer('CONCAT')(conv_3, up_3)
        up_2, _ = block_layer(concat_3, is_training)
        print block_layer

        block_layer = UNetBlock('UPSAMPLE',
                                (self.n_features[2], self.n_features[2]),
                                (3, 3), with_downsample_branch=False,
                                name='R2')
        concat_2 = ElementwiseLayer('CONCAT')(conv_2, up_2)
        up_1, _ = block_layer(concat_2, is_training)
        print block_layer

        block_layer = UNetBlock('NONE',
                                (self.n_features[1], self.n_features[1], self.num_classes),
                                (3, 3, 1), with_downsample_branch=True,
                                name='R1_FC')
        concat_1 = ElementwiseLayer('CONCAT')(conv_1, up_1)
        # note for the last layer, upsampling path is not used
        _, output_tensor = block_layer(concat_1, is_training)
        print block_layer
        return output_tensor


SUPPORTED_OP = set(['DOWNSAMPLE', 'UPSAMPLE', 'NONE'])
class UNetBlock(Layer):
    def __init__(self,
                 func,
                 n_chns,
                 kernels,
                 with_downsample_branch=False,
                 acti_type='relu',
                 name='UNet_block'):
        self.func = func.upper()
        assert(self.func in SUPPORTED_OP)

        super(UNetBlock, self).__init__(name=name)
        self.kernels = kernels
        self.n_chns = n_chns
        self.with_downsample_branch = with_downsample_branch
        self.acti_type = acti_type

    def layer_op(self, input_tensor, is_training):
        output_tensor = input_tensor
        for (kernel_size, n_features) in zip(self.kernels, self.n_chns):
            conv_op = ConvolutionalLayer(n_output_chns=n_features,
                                         kernel_size=kernel_size,
                                         acti_fun=self.acti_type,
                                         name='{}'.format(n_features))
            output_tensor = conv_op(output_tensor, is_training)

        if self.with_downsample_branch:
            branch_output = output_tensor
        else:
            branch_output = None

        if self.func == 'DOWNSAMPLE':
            downsample_op = DownSampleLayer('MAX',
                                            kernel_size=2,
                                            stride=2,
                                            name='down_2x2')
            output_tensor = downsample_op(output_tensor)
        elif self.func == 'UPSAMPLE':
            upsample_op = DeconvolutionalLayer(n_output_chns=self.n_chns[-1],
                                               kernel_size=2,
                                               stride=2,
                                               name='up_2x2')
            output_tensor = upsample_op(output_tensor, is_training)
        elif self.func == 'NONE':
            pass # do nothing
        return output_tensor, branch_output
