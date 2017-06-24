# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from layer.base_layer import TrainableLayer
from layer.reshape import ReshapeLayer
from layer.fully_connected import FullyConnectedLayer
from layer.layer_util import infer_dims
from layer.convolution import ConvolutionalLayer
from layer.convolution_transpose import ConvolutionalTransposeLayer
from layer.convolution_transpose_unpooling import ConvolutionalTransposeUnpoolingLayer

class AE_convolutional(TrainableLayer):
    """
        This is the most basic autoencoder, composed of a sequence of
        fully-connected layers.
        """

    def __init__(self,
                 num_classes=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='AE_convolutional'):

        super(AE_convolutional, self).__init__(name=name)
        self.conv_features = [20, 30]
        self.acti_func_conv = ['relu', 'relu']
        self.trans_conv_features = [30, 20]
        self.acti_func_trans_conv = ['relu', 'sigmoid']
        self.layer_sizes_encoder = [256, 128]
        self.acti_func_encoder = ['relu', 'relu']
        self.layer_sizes_decoder = [256]
        self.acti_func_decoder = ['relu']
        self.acti_func_output = 'relu'

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, images, is_training):

        [data_dimensions, data_dimensionality] = infer_dims(images)
        # Assume for now we always follow a CNN encoder with 2x2x2 downsampling
        data_downsampled_dimensions = data_dimensions
        for p in range(0,len(data_downsampled_dimensions)-1):
            data_downsampled_dimensions[p] /= 2**len(self.conv_features)
            data_downsampled_dimensions[p] = int(data_downsampled_dimensions[p])
        data_downsampled_dimensions[-1] = self.conv_features[-1]
        data_downsampled_dimensionality = int(data_dimensionality / (2**(3*len(self.conv_features))))

        # Define the encoding convolution layers
        encoders_cnn = []
        encoders_downsamplers = []
        for p in range(0,len(self.conv_features)):
            encoders_cnn.append(ConvolutionalLayer(
                n_output_chns=self.conv_features[p],
                kernel_size=3,
                padding='SAME',
                with_bias=True,
                with_bn=True,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func=self.acti_func_conv[p],
                name='encoding_conv_'+str(p)))
            print(encoders_cnn[-1])

            encoders_downsamplers.append(ConvolutionalLayer(
                n_output_chns=self.conv_features[p],
                kernel_size=2,
                stride = 2,
                padding='SAME',
                with_bias=False,
                with_bn=False,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func='identity',
                name='encoding_downsampler_'+str(p)))
            print(encoders_downsamplers[-1])

        reshape_downsampled_data = ReshapeLayer([-1, data_downsampled_dimensionality*self.conv_features[-1]])
        print(reshape_downsampled_data)

        # Define the encoding layers
        encoders = []
        for i in range(0,len(self.layer_sizes_encoder)):
            encoders.append(FullyConnectedLayer(
                n_output_chns=self.layer_sizes_encoder[i],
                acti_func=self.acti_func_encoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='fc_encoder_{}'.format(self.layer_sizes_encoder[i])))
            print(encoders[-1])

        # Define the hidden decoding layers
        decoders = []
        for i in range(0, len(self.layer_sizes_decoder)):
            decoders.append(FullyConnectedLayer(
                n_output_chns=self.layer_sizes_decoder[i],
                acti_func=self.acti_func_decoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='fc_decoder_{}'.format(self.layer_sizes_decoder[i])))
            print(decoders[-1])

        # Define the output layer
        decoders.append(FullyConnectedLayer(
            n_output_chns=data_downsampled_dimensionality*self.conv_features[-1],
            acti_func=self.acti_func_output,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_decoder_{}'.format(data_dimensionality)))
        print(decoders[-1])

        reshape_output = ReshapeLayer([-1]+data_downsampled_dimensions)
        print(reshape_output)

        # Define the decoding convolution layers
        decoders_cnn = []
        decoders_upsamplers = []
        for p in range(0, len(self.trans_conv_features)):
            decoders_upsamplers.append(ConvolutionalTransposeUnpoolingLayer(
                n_output_chns=self.trans_conv_features[p],
                kernel_size=2,
                stride=2,
                padding='SAME',
                with_bias=False,
                with_bn=False,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func='identity',
                name='upsampler_' + str(p)))
            print(decoders_upsamplers[-1])

            decoders_cnn.append(ConvolutionalTransposeLayer(
                n_output_chns=self.trans_conv_features[p],
                kernel_size=3,
                padding='SAME',
                with_bias=True,
                with_bn=True,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func=self.acti_func_trans_conv[p],
                name='conv_trans_' + str(p)))
            print(decoders_cnn[-1])

        flow = images
        for i in range(0, len(self.conv_features)):
            flow = encoders_cnn[i](flow, is_training)
            flow = encoders_downsamplers[i](flow, is_training)
        flow = reshape_downsampled_data(flow)
        for i in range(0, len(self.layer_sizes_encoder)):
            flow = encoders[i](flow, is_training)
        codes = flow
        for i in range(0, len(self.layer_sizes_decoder)+1):
            flow = decoders[i](flow, is_training)
        flow = reshape_output(flow)
        for i in range(0, len(self.trans_conv_features)):
            flow = decoders_upsamplers[i](flow, is_training)
            flow = decoders_cnn[i](flow, is_training)
        reconstructions = flow

        return [images, reconstructions, codes]
