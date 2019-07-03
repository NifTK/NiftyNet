# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer import layer_util
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.crop import CropLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.upsample import UpSampleLayer
from niftynet.network.base_net import BaseNet


class DeepMedic(BaseNet):
    """
    ### Description
    reimplementation of DeepMedic:
        Kamnitsas et al., "Efficient multi-scale 3D CNN with fully connected
        CRF for accurate brain lesion segmentation", MedIA '17

    ### Building blocks
    [CONV]          - 3x3x3 convolutional layer
    [denseCONV]     - 1x1x1 convolutional layer

    ### Diagram
    INPUT --> CROP -------> [CONV]x8 ------> [SUM] ----> [denseCONV]x3 --> OUTPUT
                |                             |
            DOWNSAMPLE ---> [CONV]x8 ---> UPSAMPLE


    ### Constraints:
    - The downsampling factor (d_factor) should be odd
    - Label size = [(image_size / d_factor) - 16]* d_factor
    - Image size should be divisible by d_factor

    # Examples:
    - Appropriate configuration for training:
    image spatial window size = 57, label spatial window size = 9, d_ factor = 3
    - Appropriate configuration for inference:
    image spatial window size = 105, label spatial window size = 57, d_ factor = 3
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name="DeepMedic"):
        """

        :param num_classes: int, number of channels of output
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param acti_func: activation function to use
        :param name: layer name
        """

        super(DeepMedic, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.d_factor = 3  # downsampling factor - should be odd
        self.crop_diff = ((self.d_factor - 1) * 16) // 2
        self.conv_features = [30, 30, 40, 40, 40, 40, 50, 50]
        self.fc_features = [150, 150, num_classes]

    def layer_op(self, images, is_training, layer_id=-1, **unused_kwargs):
        """

        :param images: tensor, input to the network, size should be divisible by d_factor
        :param is_training: boolean, True if network is in training mode
        :param layer_id: not in use
        :param unused_kwargs:
        :return: tensor, network output
        """
        # image_size is defined as the largest context, then:
        #   downsampled path size: image_size / d_factor
        #   downsampled path output: image_size / d_factor - 16

        # to make sure same size of feature maps from both pathways:
        #   normal path size: (image_size / d_factor - 16) * d_factor + 16
        #   normal path output: (image_size / d_factor - 16) * d_factor

        # where 16 is fixed by the receptive field of conv layers
        # TODO: make sure label_size = image_size/d_factor - 16

        # image_size has to be an odd number and divisible by 3 and
        # smaller than the smallest image size of the input volumes

        # label_size should be (image_size/d_factor - 16) * d_factor

        assert self.d_factor % 2 == 1  # to make the downsampling centered
        assert (layer_util.check_spatial_dims(
            images, lambda x: x % self.d_factor == 0))
        assert (layer_util.check_spatial_dims(
            images, lambda x: x % 2 == 1))  # to make the crop centered
        assert (layer_util.check_spatial_dims(
            images,
            lambda x: x > self.d_factor * 16))  # required by receptive field

        # crop 25x25x25 from 57x57x57
        crop_op = CropLayer(border=self.crop_diff, name='cropping_input')
        normal_path = crop_op(images)
        print(crop_op)

        # downsample 19x19x19 from 57x57x57
        downsample_op = DownSampleLayer(func='CONSTANT',
                                        kernel_size=self.d_factor,
                                        stride=self.d_factor,
                                        padding='VALID',
                                        name='downsample_input')
        downsample_path = downsample_op(images)
        print(downsample_op)

        # convolutions for both pathways
        for n_features in self.conv_features:
            # normal pathway convolutions
            conv_path_1 = ConvolutionalLayer(
                n_output_chns=n_features,
                kernel_size=3,
                padding='VALID',
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func=self.acti_func,
                name='normal_conv')
            normal_path = conv_path_1(normal_path, is_training)
            print(conv_path_1)

            # downsampled pathway convolutions
            conv_path_2 = ConvolutionalLayer(
                n_output_chns=n_features,
                kernel_size=3,
                padding='VALID',
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func=self.acti_func,
                name='downsample_conv')
            downsample_path = conv_path_2(downsample_path, is_training)
            print(conv_path_2)

        # upsampling the downsampled pathway
        downsample_path = UpSampleLayer('REPLICATE',
                                        kernel_size=self.d_factor,
                                        stride=self.d_factor)(downsample_path)

        # concatenate both pathways
        output_tensor = ElementwiseLayer('CONCAT')(normal_path, downsample_path)

        # 1x1x1 convolution layer
        for n_features in self.fc_features:
            conv_fc = ConvolutionalLayer(
                n_output_chns=n_features,
                kernel_size=1,
                acti_func=self.acti_func,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='conv_1x1x1_{}'.format(n_features))
            output_tensor = conv_fc(output_tensor, is_training)
            print(conv_fc)

        return output_tensor
