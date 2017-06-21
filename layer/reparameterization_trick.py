# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from .base_layer import Layer


def noise_like(input, distribution):
    if distribution == 'Gaussian':
        output = tf.random_normal(tf.shape(input), 0.0, 1.0)
    elif distribution == 'Uniform':
        output = tf.random_uniform(tf.shape(input), minval=0.0, maxval=1.0)
    elif distribution == 'Bernoulli':
        uniform_sample = tf.random_uniform(tf.shape(input), minval=0.0, maxval=1.0)
        output = tf.where(uniform_sample > 0.5, tf.zeros_like(input), tf.ones_like(input))
    else:
        print("Unrecognised noise!")
        quit()
    return output


class ReparameterizationLayer(Layer):
    """
    This class defines a reparameterization layer, for generating approximate samples from the posterior.
    See Auto-Encoding Varitaional Bayes, Kingma & Welling, 2014
    """

    def __init__(self,
                 prior='Gaussian',
                 number_of_samples=1,
                 name='reparameterization'):
        super(ReparameterizationLayer, self).__init__(name=name)
        self.prior = prior
        self.number_of_samples = number_of_samples

    def layer_op(self, distribution_parameters):

        [means, logvariances] = distribution_parameters

        if self.number_of_samples == 1:
            noise_sample = noise_like(means, self.prior)
        else:
            stochastic_parts = []
            for p in range(0, self.number_of_samples):
                stochastic_parts.append(noise_like(means, self.prior))
            noise_sample = tf.reduce_mean(tf.stack(stochastic_parts), axis=0)

        output_tensor = means + tf.exp(0.5 * logvariances) * noise_sample

        return output_tensor

