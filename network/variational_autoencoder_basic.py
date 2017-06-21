# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from layer.base_layer import TrainableLayer
from layer.reshape import ReshapeLayer
from layer.fully_connected import FullyConnectedLayer
from layer.reparameterization_trick import ReparameterizationLayer
from layer.layer_util import infer_dims

class VAE_basic(TrainableLayer):
    """
        This is the most basic implementation of a variational autoencoder (VAE).
            See Kingma & Welling, 2014, Auto-Encoding Varitaional Bayes
        """

    def __init__(self,
                 num_classes=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='identity',
                 name='VAE_basic'):

        super(VAE_basic, self).__init__(name=name)
        self.latent_variables = 128
        self.number_of_samples_from_posterior = 1
        self.acti_func = acti_func

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, images, is_training):

        [data_dimensions, data_dimensionality] = infer_dims(images)

        reshape_input = ReshapeLayer([-1, data_dimensionality])
        print(reshape_input)

        encoder_means = FullyConnectedLayer(
            n_output_chns=self.latent_variables,
            with_bn=False,
            acti_func='none',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_encoder_means_{}'.format(self.latent_variables))
        print(encoder_means)

        encoder_logvariances = FullyConnectedLayer(
            n_output_chns=self.latent_variables,
            with_bn=False,
            acti_func='none',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_encoder_logvariances_{}'.format(self.latent_variables))
        print(encoder_logvariances)

        sampler_from_posterior = ReparameterizationLayer(
            prior='Gaussian',
            number_of_samples=self.number_of_samples_from_posterior,
            name='reparameterization_{}'.format(self.latent_variables))
        print(sampler_from_posterior)

        decoder_means = FullyConnectedLayer(
            n_output_chns=data_dimensionality,
            acti_func='sigmoid',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_decoder_logvariances_{}'.format(self.latent_variables))
        print(decoder_means)

        decoder_logvariances = FullyConnectedLayer(
            n_output_chns=data_dimensionality,
            acti_func='none',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_decoder_logvariances_{}'.format(self.latent_variables))
        print(decoder_logvariances)

        reshape_output = ReshapeLayer([-1]+data_dimensions)
        print(reshape_output)

        originals = images
        images = reshape_input(images)

        posterior_means = encoder_means(images, is_training)
        posterior_logvariances = encoder_logvariances(images, is_training)

        sample_from_posterior = sampler_from_posterior([posterior_means, posterior_logvariances])

        data_means = decoder_means(sample_from_posterior, is_training)
        data_logvariances = decoder_logvariances(sample_from_posterior, is_training)

        data_means = reshape_output(data_means)
        data_logvariances = reshape_output(data_logvariances)

        return [posterior_means, posterior_logvariances, data_means, data_logvariances, originals]
