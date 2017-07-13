# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.reshape import ReshapeLayer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.layer.layer_util import infer_dims
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.upsample import UpSampleLayer
from niftynet.layer.reparameterization_trick import ReparameterizationLayer
import niftynet.engine.logging
import tensorflow as tf
import numpy as np


class VAE(TrainableLayer):
    """
        This is a convolutional autoencoder, composed of a sequence of {convolutions & down-sampling} blocks,
        followed by a sequence of fully-connected layers, followed by a sequence of
        {transpose convolutions & up-sampling} blocks.

        2DO: make the use of FC/convolutions optional, so a fully connected AE, and a fully convolutional
        AE, are both possible.
        2DO: make the pooling sizes vectors, so we can pool in x & y but not z, say.
        2DO: make the kernel sizes vectors, so we can have greater reach along certain axes.
        2DO: add a denoising option
        """

    def __init__(self,
                 num_classes=None,
                 acti_func=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 conv_features=None,
                 conv_kernel_sizes=None,
                 conv_pooling_factors=None,
                 acti_func_conv=None,
                 layer_sizes_encoder=None,
                 acti_func_encoder=None,
                 layer_sizes_decoder=None,
                 acti_func_decoder=None,
                 acti_func_fully_connected_output=None,
                 trans_conv_features=None,
                 trans_conv_kernels_sizes=None,
                 trans_conv_unpooling_factors=None,
                 upsampling_mode=None,
                 acti_func_trans_conv_means=None,
                 acti_func_trans_conv_logvariances=None,
                 name='VAE'):

        super(VAE, self).__init__(name=name)

        self.number_of_latent_variables = 512
        self.number_of_samples_from_posterior_per_example = 1

        # Set the bound on logvariance, to keep variance = exp(logvariance) within reasonable limits:
        self.logvariance_upper_bound = 40
        self.logvariance_lower_bound = -40

        # Specify the encoder here
        if conv_features == None:
            self.conv_features = [20, 40, 60]
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

        if layer_sizes_decoder == None:
            self.layer_sizes_decoder = [512]
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
            self.trans_conv_features = [40, 20, 1]
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
        if acti_func_trans_conv_means == None:
            self.acti_func_trans_conv_means = ['relu', 'relu', 'sigmoid']
        else:
            self.acti_func_trans_conv_means = acti_func_trans_conv_means
        if acti_func_trans_conv_logvariances == None:
            self.acti_func_trans_conv_logvariances = ['relu', 'relu', None]
        else:
            self.acti_func_trans_conv_logvariances = acti_func_trans_conv_logvariances

        # Choose how to upsample the feature maps in the convolutional decoding layers. Options are:
        # 1. 'DECONV' (recommended) i.e., kernel shape is HxWxDxInxOut,
        # 2. 'CHANNELWISE_DECONV' i.e., kernel shape is HxWxDx1x1,
        # 3. 'REPLICATE' i.e., no parameters
        if upsampling_mode == None:
            self.upsampling_mode = 'DECONV'
        else:
            self.upsampling_mode = upsampling_mode

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, images, is_training):

        def clip(input):
            # We will clip logvars, so that variances = exp(logvars) is well-behaved
            output = tf.maximum(input, self.logvariance_lower_bound)
            output = tf.minimum(output, self.logvariance_upper_bound)
            return output

        # Calculate the dimensionality of the data as it emerges from the convolutional part of the encoder
        # NB: we assume for now that we follow each convolutional encoder layer with 2x2x2 downsampling
        [data_dims, data_dims_product] = infer_dims(images)
        data_downsampled_dimensions = data_dims.copy()
        for p in range(0, len(data_downsampled_dimensions) - 1):
            data_downsampled_dimensions[p] /= 2 ** len(self.conv_features)
            data_downsampled_dimensions[p] = int(data_downsampled_dimensions[p])
        data_downsampled_dimensions[-1] = self.conv_features[-1]
        data_downsampled_dimensionality = int(data_dims_product / (2 ** (3 * len(self.conv_features))))

        number_of_input_channels = data_dims[-1]
        assert (self.trans_conv_features[-1] == number_of_input_channels), \
            "The number of output channels must match the number of input channels"

        # Define the encoding convolution layers
        encoders_cnn = []
        encoders_downsamplers = []
        for i in range(0, len(self.conv_features)):
            encoders_cnn.append(ConvolutionalLayer(
                n_output_chns=self.conv_features[i],
                kernel_size=self.conv_kernel_sizes[i],
                padding='SAME',
                with_bias=True,
                with_bn=True,
                w_initializer=self.initializers['w'],
                w_regularizer=None,
                acti_func=self.acti_func_conv[i],
                name='encoder_conv_{}_{}'.format(self.conv_kernel_sizes[i], self.conv_features[i])))
            print(encoders_cnn[-1])

            encoders_downsamplers.append(ConvolutionalLayer(
                n_output_chns=self.conv_features[i],
                kernel_size=2,
                stride=2,
                padding='SAME',
                with_bias=False,
                with_bn=False,
                w_initializer=self.initializers['w'],
                w_regularizer=None,
                acti_func=None,
                name='encoder_downsampler_{}_{}'.format(2, 2)))
            print(encoders_downsamplers[-1])

        raster_feature_maps = ReshapeLayer([-1, data_downsampled_dimensionality * self.conv_features[-1]])
        print(raster_feature_maps)

        # Define the encoding fully connected layers
        encoders_fc = []
        for i in range(0, len(self.layer_sizes_encoder)):
            encoders_fc.append(FullyConnectedLayer(
                n_output_nodes=self.layer_sizes_encoder[i],
                with_bias=True,
                with_bn=True,
                acti_func=self.acti_func_encoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='encoder_fc_{}'.format(self.layer_sizes_encoder[i])))
            print(encoders_fc[-1])

        encoder_means = FullyConnectedLayer(
            n_output_nodes=self.number_of_latent_variables,
            with_bn=False,
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='encoder_fc_means_{}'.format(self.number_of_latent_variables))
        print(encoder_means)

        encoder_logvariances = FullyConnectedLayer(
            n_output_nodes=self.number_of_latent_variables,
            with_bn=False,
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='encoder_fc_logvars_{}'.format(self.number_of_latent_variables))
        print(encoder_logvariances)

        sampler_from_posterior = ReparameterizationLayer(
            prior='Gaussian',
            number_of_samples=self.number_of_samples_from_posterior_per_example,
            name='reparameterization_{}'.format(self.number_of_latent_variables))
        print(sampler_from_posterior)

        # Define the decoding fully connected layers
        decoders_fc = []
        for i in range(0, len(self.layer_sizes_decoder)):
            decoders_fc.append(FullyConnectedLayer(
                n_output_nodes=self.layer_sizes_decoder[i],
                with_bias=True,
                with_bn=True,
                acti_func=self.acti_func_decoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='decoder_fc_{}'.format(self.layer_sizes_decoder[i])))
            print(decoders_fc[-1])

        # Define the final decoding fully connected layer
        decoders_fc.append(FullyConnectedLayer(
            n_output_nodes=data_downsampled_dimensionality * self.conv_features[-1],
            with_bias=True,
            with_bn=True,
            acti_func=self.acti_func_fully_connected_output,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='decoder_fc_{}'.format(data_downsampled_dimensionality * self.conv_features[-1])))
        print(decoders_fc[-1])

        unraster_feature_maps = ReshapeLayer([-1] + data_downsampled_dimensions)
        print(unraster_feature_maps)

        # Define the decoding convolution layers
        decoders_means_cnn = []
        decoders_means_upsamplers = []
        decoders_logvars_cnn = []
        decoders_logvars_upsamplers = []
        for i in range(0, len(self.trans_conv_features)):
            if self.upsampling_mode == 'DECONV':
                decoders_means_upsamplers.append(DeconvolutionalLayer(
                    n_output_chns=self.trans_conv_features[i],
                    kernel_size=2,
                    stride=2,
                    padding='SAME',
                    with_bias=True,
                    with_bn=True,
                    w_initializer=self.initializers['w'],
                    w_regularizer=None,
                    acti_func=None,
                    name='decoder_upsampler_means_{}_{}'.format(2, 2)))
                print(decoders_means_upsamplers[-1])

            decoders_means_cnn.append(DeconvolutionalLayer(
                n_output_chns=self.trans_conv_features[i],
                kernel_size=self.trans_conv_kernels_sizes[i],
                stride=1,
                padding='SAME',
                with_bias=True,
                with_bn=not (i == len(self.trans_conv_features) - 1),  # No BN on output
                w_initializer=self.initializers['w'],
                w_regularizer=None,
                acti_func=self.acti_func_trans_conv_means[i],
                name='decoder_trans_conv_means_{}_{}'.format(self.trans_conv_kernels_sizes[i],
                                                             self.trans_conv_features[i])))
            print(decoders_means_cnn[-1])

        for i in range(0, len(self.trans_conv_features)):
            if self.upsampling_mode == 'DECONV':
                decoders_logvars_upsamplers.append(DeconvolutionalLayer(
                    n_output_chns=self.trans_conv_features[i],
                    kernel_size=2,
                    stride=2,
                    padding='SAME',
                    with_bias=True,
                    with_bn=True,
                    w_initializer=self.initializers['w'],
                    w_regularizer=None,
                    acti_func=None,
                    name='decoder_upsampler_variances_{}_{}'.format(2, 2)))
                print(decoders_logvars_upsamplers[-1])

            decoders_logvars_cnn.append(DeconvolutionalLayer(
                n_output_chns=self.trans_conv_features[i],
                kernel_size=self.trans_conv_kernels_sizes[i],
                stride=1,
                padding='SAME',
                with_bias=True,
                with_bn=not (i == len(self.trans_conv_features) - 1),  # No BN on output
                w_initializer=self.initializers['w'],
                w_regularizer=None,
                acti_func=self.acti_func_trans_conv_logvariances[i],
                name='decoder_trans_conv_variances_{}_{}'.format(self.trans_conv_kernels_sizes[i],
                                                                 self.trans_conv_features[i])))
            print(decoders_logvars_cnn[-1])

        # Convolutional encoder layers
        flow = images
        for i in range(0, len(self.conv_features)):
            flow = encoders_cnn[i](flow, is_training)
            flow = encoders_downsamplers[i](flow, is_training)
        flow = raster_feature_maps(flow)

        # Fully connected encoder layers
        for i in range(0, len(self.layer_sizes_encoder)):
            flow = encoders_fc[i](flow, is_training)

        posterior_means = encoder_means(flow, is_training)
        posterior_logvars = clip(encoder_logvariances(flow, is_training))
        # We want access to the codes later...
        codes = sampler_from_posterior([posterior_means, posterior_logvars])
        flow = codes

        # Fully connected decoder layers
        for i in range(0, len(self.layer_sizes_decoder) + 1):
            flow = decoders_fc[i](flow, is_training)
        # Reshape the flow into HxWxDx(ChannelsIn)x(ChannelsOut) format
        flow = unraster_feature_maps(flow)

        # We up-sample means and variances separately from the output of the fully-connected layers
        flow_means = flow
        flow_logvars = flow

        # Convolutional decoder layers, for predicting mu in the Gaussian model of the input
        for i in range(0, len(self.trans_conv_features)):
            if self.upsampling_mode == 'DECONV':
                flow_means = decoders_means_upsamplers[i](flow_means, is_training)
            elif self.upsampling_mode == 'CHANNELWISE_DECONV':
                flow_means = UpSampleLayer('CHANNELWISE_DECONV',
                                           kernel_size=2,
                                           stride=2)(flow_means)
            elif self.upsampling_mode == 'REPLICATE':
                flow_means = UpSampleLayer('REPLICATE',
                                           kernel_size=2,
                                           stride=2)(flow_means)
            flow_means = decoders_means_cnn[i](flow_means, is_training)

        # Convolutional decoder layers, for predicting (log) variance in the Gaussian model of the input
        for i in range(0, len(self.trans_conv_features)):
            if self.upsampling_mode == 'DECONV':
                flow_logvars = decoders_logvars_upsamplers[i](flow_logvars, is_training)
            elif self.upsampling_mode == 'CHANNELWISE_DECONV':
                flow_logvars = UpSampleLayer('CHANNELWISE_DECONV',
                                                  kernel_size=2,
                                                  stride=2)(flow_logvars)
            elif self.upsampling_mode == 'REPLICATE':
                flow_logvars = UpSampleLayer('REPLICATE',
                                                  kernel_size=2,
                                                  stride=2)(flow_logvars)
            flow_logvars = decoders_logvars_cnn[i](flow_logvars, is_training)

        data_logvars = clip(flow_logvars)

        data_variances = tf.exp(data_logvars)
        posterior_variances = tf.exp(posterior_logvars)

        # Monitor the KL divergence of the (approximate) posterior from the prior
        KL_divergence = 1 + posterior_logvars - tf.square(posterior_means) - tf.exp(posterior_logvars)
        KL_divergence = -0.5 * tf.reduce_mean(tf.reduce_sum(KL_divergence, axis=[1]))
        KL = tf.summary.scalar('KL_divergence', KL_divergence)
        tf.add_to_collection(niftynet.engine.logging.CONSOLE, KL)

        # Monitor the negative log likelihood of the parameters given the training data
        log_likelihood = data_logvars + np.log(2 * np.pi) + tf.exp(-data_logvars) * tf.square(flow_means - images)
        log_likelihood = -0.5 * tf.reduce_mean(tf.reduce_sum(log_likelihood, axis=[1, 2, 3, 4]))
        NLL = tf.summary.scalar('negative_log_likelihood', -log_likelihood)
        tf.add_to_collection(niftynet.engine.logging.CONSOLE, NLL)

        # Send (animated!) 3D reconstructions to TensorBoard
        def normalise(input):
            min = tf.reduce_min(input)
            max = tf.reduce_max(input)
            return 255 * (input-min) / (max-min)
        niftynet.engine.logging.image3_coronal('Originals', normalise(images))
        niftynet.engine.logging.image3_coronal('Means', normalise(flow_means))
        niftynet.engine.logging.image3_coronal('Variances', normalise(data_variances))

        return [posterior_means, posterior_logvars, flow_means, data_logvars,
                images, data_variances, posterior_variances]
