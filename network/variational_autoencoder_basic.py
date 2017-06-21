# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from layer.base_layer import TrainableLayer
from layer.reshape import ReshapeLayer
from layer.fully_connected import FullyConnectedLayer
from layer.reparameterization_trick import ReparameterizationLayer
from layer.layer_util import infer_dims

import tensorflow as tf

class VAE_basic(TrainableLayer):
    # """
    #     This is the most basic implementation of a variational autoencoder (VAE).
    #     See Kingma & Welling, 2014, Auto-Encoding Varitaional Bayes
    #     Download the archive https://www.dropbox.com/s/6kvqyxk73pooq7r/IXI_reprocessed.tar?dl=0 and place in the
    #     example_volumes folder in the NiftyNet base directory.
    #     run python run_application train -c C:\Users\rober\Documents\NiftyNet\config\variational_autoencoder_config.txt
    #     """

    def __init__(self,
                 num_classes=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='identity',
                 name='VAE_basic'):

        super(VAE_basic, self).__init__(name=name)
        self.layer_sizes_encoder = [256, 128] #
        self.acti_func_encoder = ['relu', 'relu']
        self.number_of_latent_variables = 32
        self.number_of_samples_from_posterior_per_example = 1
        self.layer_sizes_decoder = [128, 256]
        self.acti_func_decoder = ['relu', 'relu']
        self.acti_func_output_means = 'sigmoid'
        self.acti_func_output_logvariances = 'identity'
        self.logvariance_upper_bound = 80 # For x as little as 100, exp(x) = 2.6 x 10^43, so must bound this above.
        self.logvariance_lower_bound = -80 # As variance --> 0, logvariance --> -inf, so must bound this below.

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, images, is_training):

        [data_dimensions, data_dimensionality] = infer_dims(images)

        reshape_input = ReshapeLayer([-1, data_dimensionality])
        print(reshape_input)

        # Define the encoding layers
        encoders = []
        for i in range(0, len(self.layer_sizes_encoder)):
            encoders.append(FullyConnectedLayer(
                n_output_chns=self.layer_sizes_encoder[i],
                with_bn=True,
                acti_func=self.acti_func_encoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='fc_encoder_{}'.format(self.layer_sizes_encoder[i])))
            print(encoders[-1])

        encoder_means = FullyConnectedLayer(
            n_output_chns=self.number_of_latent_variables,
            with_bn=False,
            acti_func='identity',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_encoder_means_{}'.format(self.number_of_latent_variables))
        print(encoder_means)

        encoder_logvariances = FullyConnectedLayer(
            n_output_chns=self.number_of_latent_variables,
            with_bn=False,
            acti_func='identity',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_encoder_logvariances_{}'.format(self.number_of_latent_variables))
        print(encoder_logvariances)

        sampler_from_posterior = ReparameterizationLayer(
            prior='Gaussian',
            number_of_samples=self.number_of_samples_from_posterior_per_example,
            name='reparameterization_{}'.format(self.number_of_latent_variables))
        print(sampler_from_posterior)

        # Define the hidden decoding layers
        decoders = []
        for i in range(0, len(self.layer_sizes_decoder)):
            decoders.append(FullyConnectedLayer(
                n_output_chns=self.layer_sizes_decoder[i],
                with_bn=True,
                acti_func=self.acti_func_decoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='fc_decoder_{}'.format(self.layer_sizes_decoder[i])))
            print(decoders[-1])

        decoder_means = FullyConnectedLayer(
            n_output_chns=data_dimensionality,
            with_bn=False,
            acti_func=self.acti_func_output_means,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_decoder_means_{}'.format(data_dimensionality))
        print(decoder_means)

        decoder_logvariances = FullyConnectedLayer(
            n_output_chns=data_dimensionality,
            with_bn=False,
            acti_func=self.acti_func_output_logvariances,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_decoder_logvariances_{}'.format(data_dimensionality))
        print(decoder_logvariances)

        reshape_output = ReshapeLayer([-1]+data_dimensions)
        print(reshape_output)

        flow = reshape_input(images)
        for i in range(0, len(self.layer_sizes_encoder)):
            flow = encoders[i](flow, is_training)

        posterior_means = encoder_means(flow, is_training)
        posterior_logvariances = encoder_logvariances(flow, is_training)

        posterior_logvariances = tf.maximum(posterior_logvariances, self.logvariance_lower_bound)
        posterior_logvariances = tf.minimum(posterior_logvariances, self.logvariance_upper_bound)

        flow = sampler_from_posterior([posterior_means, posterior_logvariances])
        for i in range(0, len(self.layer_sizes_decoder)):
            flow = decoders[i](flow, is_training)

        data_means = decoder_means(flow, is_training)
        data_means = reshape_output(data_means)
        data_logvariances = decoder_logvariances(flow, is_training)
        data_logvariances = reshape_output(data_logvariances)

        data_variances = tf.exp(data_logvariances)
        posterior_variances = tf.exp(posterior_logvariances)

        data_logvariances = tf.maximum(data_logvariances, self.logvariance_lower_bound)
        data_logvariances = tf.minimum(data_logvariances, self.logvariance_upper_bound)

        return [posterior_means, posterior_logvariances, data_means, data_logvariances,
                images, data_variances, posterior_variances]
