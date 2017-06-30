# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

def QuantitiesToMonitor(predictions, labels, dictionary):
    """
    Change which quantities to calculate and monitor while training a model
    :dictionary predictions: outputs of the model.
    :dictionary labels: ground truth.
    :dictionary dictionary: the dictionary of quantities to monitor
    :return: the list of quantities to calculate and display during training
    """
    quantities_to_monitor = []
    names_of_quantities = []

    if dictionary.get('miss_rate', True):
        quantities_to_monitor.append(tf.reduce_mean(tf.cast(
            tf.not_equal(tf.argmax(predictions, -1), labels[..., 0]),
            dtype=tf.float32)))
        names_of_quantities.append('miss rate')

    if dictionary.get('KLD', False):
        poserior_means = predictions[0]
        poserior_logvariances = predictions[1]
        KL_divergence = -0.5 * tf.reduce_mean(
            1 + poserior_logvariances - tf.square(poserior_means) - tf.exp(poserior_logvariances), axis=[1])
        quantities_to_monitor.append(tf.reduce_mean(KL_divergence))
        names_of_quantities.append('KL divergence')

    if dictionary.get('negative_log_likelihood', False):
        originals = predictions[4]
        data_means = predictions[2]
        data_logvariances = predictions[3]
        squared_differences = tf.square(data_means - originals)
        log_likelihood = -0.5 * (
        data_logvariances + np.log(2 * np.pi) + tf.exp(-data_logvariances) * squared_differences)
        quantities_to_monitor.append(tf.reduce_mean(tf.reduce_sum(-log_likelihood, axis=[1, 2, 3, 4])))
        names_of_quantities.append('negative log likelihood')

    return [quantities_to_monitor, names_of_quantities]
