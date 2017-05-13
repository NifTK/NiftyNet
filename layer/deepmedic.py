# -*- coding: utf-8 -*-
from .base import Layer
from .convolution import ConvolutionalLayer
from .crop import CropLayer
from .downsample import DownSampleLayer
from .upsample import UpSampleLayer
from .elementwise import ElementwiseLayer
from . import layer_util


class DeepMedic(Layer):
    """
    reimplementation of DeepMedic:
      Kamnitsas et al., "Efficient multi-scale 3D CNN with fully connected
      CRF for accurate brain lesion segmentation", MedIA '17
    """

    def __init__(self, num_classes):
        self.layer_name = 'DeepMedic'
        super(DeepMedic, self).__init__(name=self.layer_name)
        self.d_factor = 3  # downsampling factor
        self.crop_diff = ((self.d_factor - 1) * 16) / 2
        self.conv_features = [30, 30, 40, 40, 40, 40, 50, 50]
        self.fc_features = [150, 150, num_classes]
        self.acti_type = 'relu'
        self.num_classes = num_classes

    def layer_op(self, images, is_training, layer_id=-1):
        # image_size is defined as the largest context, then:
        #   downsampled path size: image_size / d_factor
        #   downsampled path output: image_size / d_factor - 16

        # to make sure same size of feature maps from both pathways:
        #   normal path size: (image_size / d_factor - 16) * d_factor + 16
        #   normal path output: (image_size / d_factor - 16) * d_factor

        # where 16 is fixed by the receptive field of conv layers
        # TODO: make sure label_size = image_size/d_factor - 16

        assert self.d_factor % 2 == 1  # to make the downsampling centered
        assert (layer_util.check_spatial_dims(
            images, lambda x: x % self.d_factor == 0))
        assert (layer_util.check_spatial_dims(
            images, lambda x: x % 2 == 1))  # to make the crop centered
        assert (layer_util.check_spatial_dims(
            images, lambda x: x > self.d_factor * 16))  # minimum receptive field

        ### crop 25x25x25 from 57x57x57
        crop_op = CropLayer(border=self.crop_diff, name='cropping_input')
        normal_path = crop_op(images)
        print crop_op

        ### downsample 25x25x25 from 57x57x57
        downsample_op = DownSampleLayer(func='CONSTANT',
                                        kernel_size=self.d_factor,
                                        stride=self.d_factor,
                                        padding='VALID',
                                        name='downsample_input')
        downsample_path = downsample_op(images)
        print downsample_op

        ### convolutions for both pathways
        for n_features in self.conv_features:
            # normal pathway convolutions
            conv_path_1 = ConvolutionalLayer(n_output_chns=n_features,
                                             kernel_size=3,
                                             padding='VALID',
                                             acti_fun=self.acti_type,
                                             name='normal_conv')
            normal_path = conv_path_1(normal_path, is_training)
            # downsampled pathway convolutions
            conv_path_2 = ConvolutionalLayer(n_output_chns=n_features,
                                             kernel_size=3,
                                             padding='VALID',
                                             acti_fun=self.acti_type,
                                             name='downsample_conv')
            downsample_path = conv_path_2(downsample_path, is_training)
            print conv_path_1
            print conv_path_2

        ### upsampling the downsampled pathway
        downsample_path = UpSampleLayer('REPLICATE',
                                        kernel_size=self.d_factor,
                                        stride=self.d_factor)(downsample_path)

        ### concatenate both pathways
        output_tensor = ElementwiseLayer('CONCAT')(normal_path, downsample_path)

        ### 1x1x1 convolution layer
        for n_features in self.fc_features:
            conv_fc = ConvolutionalLayer(n_output_chns=n_features,
                                         kernel_size=1,
                                         acti_fun=self.acti_type,
                                         name='conv_1x1x1_{}'.format(n_features))
            output_tensor = conv_fc(output_tensor, is_training)
            print conv_fc

        return output_tensor
