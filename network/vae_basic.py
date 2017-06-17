# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from layer.base_layer import TrainableLayer
from layer.reshape import ReshapeLayer
from layer.fully_connected import FullyConnectedLayer
from layer.vae_reparameterization_trick import ReparameterizationLayer

class VAE_basic(TrainableLayer):
    """
        reimplementation of variational autoencoder (VAE):
            see Kingma & Welling, 2014, Auto-Encoding Varitaional Bayes
        """

    def __init__(self,
                 num_classes=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='VAE_basic'):

        super(VAE_basic, self).__init__(name=name)
        self.latent_variables = 20
        self.number_of_samples_from_posterior = 1
        self.acti_func = acti_func
        self.acti_func_output = 'sigmoid'
        self.data_dimensionality = 32**3

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, images, is_training):

        reshape_input = ReshapeLayer([-1,self.data_dimensionality])
        print(reshape_input)

        encoder_means = FullyConnectedLayer(
            n_output_chns=self.latent_variables,
            with_bn=False,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_encoder_means_{}'.format(self.latent_variables))
        print(encoder_means)

        encoder_logvariances = FullyConnectedLayer(
            n_output_chns=self.latent_variables,
            with_bn=False,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_encoder_logvariances_{}'.format(self.latent_variables))
        print(encoder_logvariances)

        sample_from_posterior = ReparameterizationLayer(
            prior='Gaussian',
            number_of_samples=self.number_of_samples_from_posterior,
            name='reparameterization_{}'.format(self.latent_variables))
        print(sample_from_posterior)

        decoder_means = FullyConnectedLayer(
            n_output_chns=self.data_dimensionality,
            acti_func=self.acti_func_output,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_decoder_logvariances_{}'.format(self.latent_variables))
        print(decoder_means)

        decoder_logvariances = FullyConnectedLayer(
            n_output_chns=self.data_dimensionality,
            acti_func='relu_translated_left',
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_decoder_logvariances_{}'.format(self.latent_variables))
        print(decoder_logvariances)

        reshape_output = ReshapeLayer([-1, 32, 32, 32, 1])
        print(reshape_output)

        originals = images
        images = reshape_input(images)

        posterior_means = encoder_means(images, is_training)
        posterior_logvariances = encoder_logvariances(images, is_training)

        samle_from_posterior = sample_from_posterior([posterior_means, posterior_logvariances])

        predicted_means = decoder_means(samle_from_posterior, is_training)
        predicted_logvariances = decoder_logvariances(samle_from_posterior, is_training)

        predicted_means = reshape_output(predicted_means)
        predicted_logvariances = reshape_output(predicted_logvariances)


        return [posterior_means, posterior_logvariances, predicted_means, predicted_logvariances, originals]
