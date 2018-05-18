# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.engine.application_factory import LossAutoencoderFactory
from niftynet.layer.base_layer import Layer


class LossFunction(Layer):
    def __init__(self,
                 loss_type='VariationalLowerBound',
                 loss_func_params=None,
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)
        if loss_func_params is not None:
            self._loss_func_params = loss_func_params
        else:
            self._loss_func_params = {}
        self._data_loss_func = None
        self.make_callable_loss_func(loss_type)

    def make_callable_loss_func(self, type_str):
        self._data_loss_func = LossAutoencoderFactory.create(type_str)

    def layer_op(self, prediction):
        with tf.device('/cpu:0'):
            return self._data_loss_func(prediction, **self._loss_func_params)


def variational_lower_bound(prediction):
    """
    This is the variational lower bound derived in
    Auto-Encoding Variational Bayes, Kingma & Welling, 2014

    :param prediction: [posterior_means, posterior_logvar,
        data_means, data_logvar, originals]

        posterior_means: predicted means for the posterior

        posterior_logvar: predicted log variances for the posterior
        data_means: predicted mean parameter
        for the voxels modelled as Gaussians

        data_logvar: predicted log variance parameter
        for the voxels modelled as Gaussians

        originals: the original inputs
    :return:
    """

    # log_2pi = np.log(2*np.pi)
    log_2pi = 1.837877
    assert len(prediction) >= 5, \
        "please see the returns of network/vae.py" \
        "for the prediction list format"
    posterior_means, posterior_logvar = prediction[:2]
    data_means, data_logvar = prediction[2:4]
    originals = prediction[4]

    squared_diff = tf.square(data_means - originals)
    log_likelihood = \
        data_logvar + log_2pi + tf.exp(-data_logvar) * squared_diff
    # batch_size = tf.shape(log_likelihood)[0]
    batch_size = log_likelihood.shape.as_list()[0]
    log_likelihood = tf.reshape(log_likelihood, shape=[batch_size, -1])
    log_likelihood = -0.5 * tf.reduce_sum(log_likelihood, axis=[1])

    KL_divergence = 1 + posterior_logvar \
                    - tf.square(posterior_means) \
                    - tf.exp(posterior_logvar)
    KL_divergence = -0.5 * tf.reduce_sum(KL_divergence, axis=[1])
    return tf.reduce_mean(KL_divergence - log_likelihood)
