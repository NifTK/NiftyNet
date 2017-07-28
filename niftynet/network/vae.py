# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.reshape import ReshapeLayer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.layer.layer_util import infer_dims, infer_downsampled_dimensions
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.upsample import UpSampleLayer
import niftynet.engine.logging as logging

import tensorflow as tf
import numpy as np


class VAE(TrainableLayer):
    """
        This is a denoising convolutional variational autoencoder, composed of a sequence of
        {convolutions & down-sampling} blocks, followed by a sequence of fully-connected
        layers, followed by a sequence of {transpose convolutions & up-sampling} blocks.
        See Auto-Encoding Variational Bayes, Kingma & Welling, 2014.
        2do: easier to train varieties of VAE, e.g., constant variance with MSE
        """

    def __init__(self,
                 num_classes=None,
                 acti_func=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='VAE'):

        super(VAE, self).__init__(name=name)

        # The following options completely specify the VAE.
        # 'upsampling_mode' determines how the feature maps in the decoding layers are upsampled. The options are,
        # 1. 'DECONV' (recommended): kernel shape is HxWxDxChannelsInxChannelsOut,
        # 2. 'CHANNELWISE_DECONV': kernel shape is HxWxDx1x1,
        # 3. 'REPLICATE': no parameters.

        # 1) Denoising
        self.denoising_variance = 0.001
        # 2) The convolutional layers
        self.conv_features = [15, 25, 35]
        self.conv_kernel_sizes = [3, 3, 3]
        self.conv_pooling_factors = [2, 2, 2]
        self.acti_func_conv = ['relu', 'relu', 'relu']
        # 3) The fully connected layers
        self.layer_sizes_encoder = [512]
        self.acti_func_encoder = ['relu']
        self.layer_sizes_decoder = self.layer_sizes_encoder[::-1]
        self.acti_func_decoder = self.acti_func_encoder[::-1]
        self.acti_func_fully_connected_output = 'relu'
        # 4) The transpose convolutional layers (means)
        self.trans_conv_features_means = self.conv_features[-2::-1]  # Excluding output channels
        self.trans_conv_kernels_sizes_means = self.conv_kernel_sizes[::-1]  # length = one more than #kernels
        self.trans_conv_unpooling_factors_means = self.conv_pooling_factors[::-1]  # length = one more than #kernels
        self.acti_func_trans_conv_means = ['relu', 'relu', None]  # length = one more than #kernels
        self.upsampling_mode_means = 'DECONV'
        # 5) The transpose convolutional layers (log variances)
        self.trans_conv_features_logvars = self.trans_conv_features_means
        self.trans_conv_kernels_sizes_logvars = self.trans_conv_kernels_sizes_means
        self.trans_conv_unpooling_factors_logvars = self.trans_conv_unpooling_factors_means
        self.acti_func_trans_conv_logvars = ['relu', 'relu', None]
        self.upsampling_mode_logvars = self.upsampling_mode_means
        # 6) The sampler
        self.number_of_latent_variables = 256
        self.number_of_samples_from_posterior = 10
        # 7) Training stability
        self.logvariance_upper_bound = 80
        self.logvariance_lower_bound = -self.logvariance_upper_bound

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, images, is_training):

        def clip(input):
            # This is for clipping logvars, so that variances = exp(logvars) behaves well
            output = tf.maximum(input, self.logvariance_lower_bound)
            output = tf.minimum(output, self.logvariance_upper_bound)
            return output

        def normalise(input):
            min_val = tf.reduce_min(input)
            max_val = tf.reduce_max(input)
            return 255 * (input - min_val) / (max_val - min_val)

        number_of_input_channels = (images.get_shape()[1::].as_list())[-1]
        self.trans_conv_features_means = self.trans_conv_features_means + [number_of_input_channels]
        self.trans_conv_features_logvars = self.trans_conv_features_logvars + [number_of_input_channels]
        [downsampled_dims, downsampled_dim] = infer_downsampled_dimensions(images, self.conv_features)

        encoder = ConvolutionalEncoder(self.denoising_variance,
                              self.conv_features,
                              self.conv_kernel_sizes,
                              self.conv_pooling_factors,
                              self.acti_func_conv,
                              self.layer_sizes_encoder,
                              self.acti_func_encoder,
                              downsampled_dim)

        sampler = GaussianSampler(self.number_of_latent_variables,
                                  self.number_of_samples_from_posterior,
                                  self.logvariance_upper_bound,
                                  self.logvariance_lower_bound)

        decoder_means = ConvolutionalDecoder(self.layer_sizes_decoder + [downsampled_dim * self.conv_features[-1]],
                                    self.acti_func_decoder + [self.acti_func_fully_connected_output],
                                    self.trans_conv_features_means,
                                    self.trans_conv_kernels_sizes_means,
                                    self.trans_conv_unpooling_factors_means,
                                    self.acti_func_trans_conv_means,
                                    self.upsampling_mode_means,
                                    shape_of_downsampled_feature_maps = downsampled_dims,
                                    name='ConvolutionalDecoder_means')

        decoder_logvars = ConvolutionalDecoder(self.layer_sizes_decoder + [downsampled_dim * self.conv_features[-1]],
                                      self.acti_func_decoder + [self.acti_func_fully_connected_output],
                                      self.trans_conv_features_logvars,
                                      self.trans_conv_kernels_sizes_logvars,
                                      self.trans_conv_unpooling_factors_logvars,
                                      self.acti_func_trans_conv_logvars,
                                      self.upsampling_mode_logvars,
                                      shape_of_downsampled_feature_maps = downsampled_dims,
                                      name='ConvolutionalDecoder_logvars')

        # Encode the input
        encoding = encoder(images, is_training)

        # Sample (approximately) from the posterior distribution, P(latent variables|input)
        [sample, posterior_means, posterior_logvars] = sampler(encoding, is_training)

        # Decode the samples
        [data_means, data_logvars] = [decoder_means(sample, is_training), clip(decoder_logvars(sample, is_training))]

        # Monitor the KL divergence of the (approximate) posterior from the prior
        KL_divergence = 1 + posterior_logvars - tf.square(posterior_means) - tf.exp(posterior_logvars)
        KL_divergence = -0.5 * tf.reduce_mean(tf.reduce_sum(KL_divergence, axis=[1]))
        tf.add_to_collection(logging.CONSOLE, tf.summary.scalar('KL_divergence', KL_divergence))

        # Monitor the (negative log) likelihood of the parameters given the data
        log_likelihood = data_logvars + np.log(2 * np.pi) + tf.exp(-data_logvars) * tf.square(data_means - images)
        log_likelihood = -0.5 * tf.reduce_mean(tf.reduce_sum(log_likelihood, axis=[1, 2, 3, 4]))
        tf.add_to_collection(logging.CONSOLE, tf.summary.scalar('negative_log_likelihood', -log_likelihood))

        data_variances = tf.exp(data_logvars)

        # Monitor reconstructions
        logging.image3_coronal('Originals', normalise(images))
        logging.image3_coronal('Means', normalise(data_means))
        logging.image3_coronal('Variances', normalise(data_variances))

        posterior_variances = tf.exp(posterior_logvars)

        return [posterior_means, posterior_logvars, data_means, data_logvars,
                images, data_variances, posterior_variances, sample]


class ConvolutionalEncoder(TrainableLayer):
    """
        This is a generic encoder composed of {convolutions & downsampling} blocks followed by
        fully connected layers.
        """
    def __init__(self,
                 denoising_variance,
                 conv_features,
                 conv_kernel_sizes,
                 conv_pooling_factors,
                 acti_func_conv,
                 layer_sizes_encoder,
                 acti_func_encoder,
                 downsampled_dim,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='ConvolutionalEncoder'):

        super(ConvolutionalEncoder, self).__init__(name=name)

        self.denoising_variance = denoising_variance
        self.conv_features = conv_features
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pooling_factors = conv_pooling_factors
        self.acti_func_conv = acti_func_conv
        self.layer_sizes_encoder = layer_sizes_encoder
        self.acti_func_encoder = acti_func_encoder
        self.downsampled_dim = downsampled_dim

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}


    def layer_op(self, images, is_training):

        # Define the encoding convolutional layers
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
                name='encoder_downsampler_2_2'))
            print(encoders_downsamplers[-1])

        serialise_feature_maps = ReshapeLayer([-1, self.downsampled_dim * self.conv_features[-1]])
        print(serialise_feature_maps)

        # Define the encoding fully-connected layers
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

        # Add Gaussian noise to the input
        if self.denoising_variance > 0:
            flow = images + tf.random_normal(tf.shape(images), 0.0, self.denoising_variance)
        else:
            flow = images

        # Convolutional encoder layers
        for i in range(0, len(self.conv_features)):
            flow = encoders_downsamplers[i]( encoders_cnn[i](flow, is_training), is_training)

        # Flatten the feature maps
        flow = serialise_feature_maps(flow)

        # Fully-connected encoder layers
        for i in range(0, len(self.layer_sizes_encoder)):
            flow = encoders_fc[i](flow, is_training)

        return flow


class GaussianSampler(TrainableLayer):
    """
        This predicts the mean and logvariance parameters, then generates an approximate sample from the
        posterior.
        """
    def __init__(self,
                 number_of_latent_variables,
                 number_of_samples_from_posterior,
                 logvariance_upper_bound,
                 logvariance_lower_bound,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='gaussian_sampler'):

        super(GaussianSampler, self).__init__(name=name)

        self.number_of_latent_variables = number_of_latent_variables
        self.number_of_samples = number_of_samples_from_posterior
        self.logvariance_upper_bound = logvariance_upper_bound
        self.logvariance_lower_bound = logvariance_lower_bound

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}


    def layer_op(self, codes, is_training):

        def clip(input):
            # This is for clipping logvars, so that variances = exp(logvars) behaves well
            output = tf.maximum(input, self.logvariance_lower_bound)
            output = tf.minimum(output, self.logvariance_upper_bound)
            return output

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

        # Predict the posterior distribution's parameters
        posterior_means = encoder_means(codes, is_training)
        posterior_logvars = clip(encoder_logvariances(codes, is_training))

        if self.number_of_samples == 1:
            noise_sample = tf.random_normal(tf.shape(posterior_means), 0.0, 1.0)
        else:
            sample_shape = tf.concat([tf.constant(self.number_of_samples, shape=[1, ]), tf.shape(posterior_means)], 0)
            noise_sample = tf.reduce_mean(tf.random_normal(sample_shape, 0.0, 1.0), axis=0)

        return [posterior_means + tf.exp(0.5 * posterior_logvars) * noise_sample, posterior_means, posterior_logvars]



class ConvolutionalDecoder(TrainableLayer):
    """
        This is a generic decoder composed of fully connected layers followed by {upsampling & transpose convolution}
        blocks.
        """
    def __init__(self,
                 layer_sizes_decoder,
                 acti_func_decoder,
                 trans_conv_features,
                 trans_conv_kernels_sizes,
                 trans_conv_unpooling_factors,
                 acti_func_trans_conv,
                 upsampling_mode,
                 shape_of_downsampled_feature_maps,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='ConvolutionalDecoder'):

        super(ConvolutionalDecoder, self).__init__(name=name)

        self.layer_sizes_decoder = layer_sizes_decoder
        self.acti_func_decoder = acti_func_decoder
        self.trans_conv_features = trans_conv_features
        self.trans_conv_kernels_sizes = trans_conv_kernels_sizes
        self.trans_conv_unpooling_factors = trans_conv_unpooling_factors
        self.acti_func_trans_conv = acti_func_trans_conv
        self.upsampling_mode = upsampling_mode
        self.downsampled_dims = shape_of_downsampled_feature_maps

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, codes, is_training):

        # Define the decoding fully-connected layers
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

        reconstitute_feature_maps = ReshapeLayer([-1] + self.downsampled_dims)
        print(reconstitute_feature_maps)

        # Define the decoding convolutional layers
        decoders_cnn = []
        decoders_upsamplers = []
        for i in range(0, len(self.trans_conv_features)):
            if self.upsampling_mode == 'DECONV':
                decoders_upsamplers.append(DeconvolutionalLayer(
                    n_output_chns=self.trans_conv_features[i],
                    kernel_size=2,
                    stride=2,
                    padding='SAME',
                    with_bias=True,
                    with_bn=True,
                    w_initializer=self.initializers['w'],
                    w_regularizer=None,
                    acti_func=None,
                    name='decoder_upsampler_2_2'))
                print(decoders_upsamplers[-1])

            decoders_cnn.append(DeconvolutionalLayer(
                n_output_chns=self.trans_conv_features[i],
                kernel_size=self.trans_conv_kernels_sizes[i],
                stride=1,
                padding='SAME',
                with_bias=True,
                with_bn=not (i == len(self.trans_conv_features) - 1),  # No BN on output
                w_initializer=self.initializers['w'],
                w_regularizer=None,
                acti_func=self.acti_func_trans_conv[i],
                name='decoder_trans_conv_{}_{}'.format(self.trans_conv_kernels_sizes[i], self.trans_conv_features[i])))
            print(decoders_cnn[-1])

        # Fully-connected decoder layers
        flow = codes
        for i in range(0, len(self.layer_sizes_decoder)):
            flow = decoders_fc[i](flow, is_training)

        # Reconstitute the feature maps
        flow = reconstitute_feature_maps(flow)

        # Convolutional decoder layers
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

        return flow
