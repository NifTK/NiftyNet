# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.reshape import ReshapeLayer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.layer.layer_util import infer_dims
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.upsample import UpSampleLayer

class AE_convolutional(TrainableLayer):
    """
        This is a convolutional autoencoder, composed of a sequence of CNN+Pooling layers,
        followed by a sequence of fully-connected layers, followed by a sequence of
        TRANS_CNN+Unpooling layers.
        
        NB: trans_conv_features[-1] must equal the number of channels in the input images. Will hard code this soon,
        or possibly add an 'assert'.
        2DO1: make the use of FC/convolutions optional, so a fully connected AE, and a fully convolutional
        AE, are both possible.
        2DO2: make the pooling sizes configurable, so we can pool in x & y but not z, say.
        2DO3: specify kernel sizes as a vector, so we can have greater reach along certain axes.
        2DO4: use an 'assert' to verify that the decoder's output dimensionality matches that of the input
        2DO5: add a denoising option
        """

    def __init__(self,
                 num_classes=None,
                 acti_func = None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 conv_features = None,
                 conv_kernel_sizes = None,
                 conv_pooling_factors = None,
                 acti_func_conv = None,
                 layer_sizes_encoder = None,
                 acti_func_encoder = None,
                 layer_sizes_decoder = None,
                 acti_func_decoder = None,
                 acti_func_fully_connected_output = None,
                 trans_conv_features = None,
                 trans_conv_kernels_sizes = None,
                 trans_conv_unpooling_factors = None,
                 upsampling_mode = None,
                 acti_func_trans_conv = None,
                 quantities_to_monitor = None,
                 name='AE_convolutional'):

        super(AE_convolutional, self).__init__(name=name)

        # Specify the encoder here
        if conv_features == None:
            self.conv_features = [20, 30, 40]
        else:
            self.conv_features = conv_features
        if conv_kernel_sizes == None:
            self.conv_kernel_sizes = [3, 3, 3]
        else:
            self.conv_kernel_sizes = conv_kernel_sizes
        if conv_pooling_factors == None:
            self.conv_pooling_factors = [2, 2, 2]
        else:
            self.conv_pooling_factors = conv_pooling_factors
        if acti_func_conv == None:
            self.acti_func_conv = ['relu', 'relu', 'relu']
        else:
            self.acti_func_conv = acti_func_conv
        if layer_sizes_encoder == None:
            self.layer_sizes_encoder = [512]
        else:
            self.layer_sizes_encoder = layer_sizes_encoder
        if acti_func_encoder == None:
            self.acti_func_encoder = ['relu']
        else:
            self.acti_func_encoder = acti_func_encoder
        # Specify the decoder here (it's not obligatory to mirror the encoder's layers in any way, but
        # the decoder's output dimensionality must match that of the input)
        if layer_sizes_decoder == None:
            self.layer_sizes_decoder = []
        else:
            self.layer_sizes_decoder = layer_sizes_decoder
        if acti_func_decoder == None:
            self.acti_func_decoder = ['relu']
        else:
            self.acti_func_decoder = acti_func_decoder
        if acti_func_fully_connected_output == None:
            self.acti_func_fully_connected_output = 'relu'
        else:
            self.acti_func_fully_connected_output = acti_func_fully_connected_output
        if trans_conv_features == None:
            self.trans_conv_features = [30, 20, 1]
        else:
            self.trans_conv_features = trans_conv_features
        if trans_conv_kernels_sizes == None:
            self.trans_conv_kernels_sizes = [3, 3, 3]
        else:
            self.trans_conv_kernels_sizes = trans_conv_kernels_sizes
        if trans_conv_unpooling_factors == None:
            self.trans_conv_unpooling_factors = [2, 2, 2]
        else:
            self.trans_conv_unpooling_factors = trans_conv_unpooling_factors
        if acti_func_trans_conv == None:
            self.acti_func_trans_conv = ['relu', 'relu', 'sigmoid']
        else:
            self.acti_func_trans_conv = acti_func_trans_conv

        # Choose how to upsample the feature maps in the convolutional decoding layers. Options are:
        # 1. 'DECONV' i.e. kernel shape is HxWxDxInxOut,
        # 2. 'CHANNELWISE_DECONV' i.e., kernel shape is HxWxDx1x1,
        # 3. 'REPLICATE' i.e. no parameters
        if upsampling_mode == None:
            self.upsampling_mode = 'CHANNELWISE_DECONV'
        else:
            self.upsampling_mode = upsampling_mode

        # Specify which quantities to calculate and print during training
        if quantities_to_monitor == None:
            self.quantities_to_monitor = {'miss_rate': False}
        else:
            self.quantities_to_monitor = quantities_to_monitor

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, images, is_training):

        [data_dims, data_dims_product] = infer_dims(images)
        # Calculate the dimensionality of the data as it emerges from the convolutional part of the encoder
        # NB: we assume for now that we always follow a CNN encoder with 2^N downsampling
        data_downsampled_dimensions = data_dims
        for p in range(0,len(data_downsampled_dimensions)-1):
            data_downsampled_dimensions[p] /= 2**len(self.conv_features)
            data_downsampled_dimensions[p] = int(data_downsampled_dimensions[p])
        data_downsampled_dimensions[-1] = self.conv_features[-1]
        data_downsampled_dimensionality = int(data_dims_product / (2**(3*len(self.conv_features))))

        # Define the encoding convolution layers
        encoders_cnn = []
        encoders_downsamplers = []
        for i in range(0,len(self.conv_features)):
            encoders_cnn.append(ConvolutionalLayer(
                n_output_chns=self.conv_features[i],
                kernel_size=self.conv_kernel_sizes[i],
                padding='SAME',
                with_bias=True,
                with_bn=True,
                w_initializer=self.initializers['w'],
                w_regularizer=None,
                acti_func=self.acti_func_conv[i],
                name='encoding_conv_{}_{}'.format(self.conv_kernel_sizes[i], self.conv_features[i])))
            print(encoders_cnn[-1])

            encoders_downsamplers.append(ConvolutionalLayer(
                n_output_chns=self.conv_features[i],
                kernel_size=2,
                stride = 2,
                padding='SAME',
                with_bias=False,
                with_bn=False,
                w_initializer=self.initializers['w'],
                w_regularizer=None,
                acti_func='identity',
                name='encoding_downsampler_{}_{}'.format(2, 2)))
            print(encoders_downsamplers[-1])

        raster_feature_maps = ReshapeLayer([-1, data_downsampled_dimensionality*self.conv_features[-1]])
        print(raster_feature_maps)

        # Define the encoding fully connected layers
        encoders_fc = []
        for i in range(0,len(self.layer_sizes_encoder)):
            encoders_fc.append(FullyConnectedLayer(
                n_output_chns=self.layer_sizes_encoder[i],
                with_bias=True,
                with_bn=True,
                acti_func=self.acti_func_encoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='fc_encoder_{}'.format(self.layer_sizes_encoder[i])))
            print(encoders_fc[-1])

        # Define the decoding fully connected layers
        decoders_fc = []
        for i in range(0, len(self.layer_sizes_decoder)):
            decoders_fc.append(FullyConnectedLayer(
                n_output_chns=self.layer_sizes_decoder[i],
                with_bias=True,
                with_bn=True,
                acti_func=self.acti_func_decoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='fc_decoder_{}'.format(self.layer_sizes_decoder[i])))
            print(decoders_fc[-1])

        # Define the final decoding fully connected layer
        decoders_fc.append(FullyConnectedLayer(
            n_output_chns=data_downsampled_dimensionality*self.conv_features[-1],
            with_bias=True,
            with_bn=True,
            acti_func=self.acti_func_fully_connected_output,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_decoder_{}'.format(data_dims_product)))
        print(decoders_fc[-1])

        unraster_feature_maps = ReshapeLayer([-1]+data_downsampled_dimensions)
        print(unraster_feature_maps)

        # Define the decoding convolution layers
        decoders_cnn = []
        decoders_upsamplers = []
        for i in range(0, len(self.trans_conv_features)):
            if self.upsampling_mode == 'DECONV':
                decoders_upsamplers.append(DeconvolutionalLayer(
                    n_output_chns=self.trans_conv_features[i],
                    kernel_size=2,
                    stride=2,
                    padding='SAME',
                    with_bias=False,
                    with_bn=False,
                    w_initializer=self.initializers['w'],
                    w_regularizer=None,
                    acti_func='identity',
                    name='upsampler_{}_{}'.format(2, 2)))
                print(decoders_upsamplers[-1])

            decoders_cnn.append(DeconvolutionalLayer(
                n_output_chns=self.trans_conv_features[i],
                kernel_size=self.trans_conv_kernels_sizes[i],
                stride=1,
                padding='SAME',
                with_bias=True,
                with_bn=not(i == len(self.trans_conv_features) - 1),
                w_initializer=self.initializers['w'],
                w_regularizer=None,
                acti_func=self.acti_func_trans_conv[i],
                name='decoding_trans_conv_{}_{}'.format(self.trans_conv_kernels_sizes[i], self.trans_conv_features[i])))
            print(decoders_cnn[-1])

        flow = images
        for i in range(0, len(self.conv_features)):
            flow = encoders_cnn[i](flow, is_training)
            flow = encoders_downsamplers[i](flow, is_training)
        flow = raster_feature_maps(flow)
        for i in range(0, len(self.layer_sizes_encoder)):
            flow = encoders_fc[i](flow, is_training)
        codes = flow
        for i in range(0, len(self.layer_sizes_decoder)+1):
            flow = decoders_fc[i](flow, is_training)
        flow = unraster_feature_maps(flow)
        for i in range(0, len(self.trans_conv_features)):
            if self.upsampling_mode == 'DECONV':
                flow = decoders_upsamplers[i](flow, is_training)
            elif self.upsampling_mode == 'CHANNELWISE_DECONV':
                flow = UpSampleLayer('CHANNELWISE_DECONV',
                                     kernel_size=2,
                                     stride=2)(flow)
            elif self.upsampling_mode == 'REPLICATE':
                flow = UpSampleLayer('REPLICATE',
                                     kernel_size=2,
                                     stride=2)(flow)
            flow = decoders_cnn[i](flow, is_training)
        reconstructions = flow

        return [images, reconstructions, codes]
