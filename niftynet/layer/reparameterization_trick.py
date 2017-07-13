# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from .base_layer import Layer

def noise_shaped_like(shape, distribution):
    if distribution == 'Gaussian':
        output = tf.random_normal(shape, 0.0, 1.0)
    elif distribution == 'Uniform':
        output = tf.random_uniform(shape, minval=0.0, maxval=1.0)
    elif distribution == 'Bernoulli':
        uniform_sample = tf.random_uniform(shape, minval=0.0, maxval=1.0)
        output = tf.where(uniform_sample > 0.5, tf.zeros(shape), tf.ones(shape))
    else:
        print("Unrecognised noise!")
        quit()
    return output

class ReparameterizationLayer(Layer):
    """
    This class defines a 'reparameterization layer', for generating approximate samples from the posterior of a VAE;
    see Auto-Encoding Variational Bayes, Kingma & Welling, 2014
    """

    def __init__(self,
                 prior='Gaussian',
                 number_of_samples=1,
                 name='reparameterization'):
        super(ReparameterizationLayer, self).__init__(name=name)
        self.prior = prior
        self.number_of_samples = number_of_samples

    def layer_op(self, means, logvariances):

        if self.number_of_samples == 1:
            noise_sample = noise_shaped_like(tf.shape(means), self.prior)
        else:
            shape_of_expanded_sample = tf.concat([tf.constant(self.number_of_samples, shape=[1, ]), tf.shape(means)], 0)
            noise_sample = noise_shaped_like(shape_of_expanded_sample, self.prior)
            noise_sample = tf.reduce_mean(noise_sample, axis=0)

        return means + tf.exp(0.5 * logvariances) * noise_sample
