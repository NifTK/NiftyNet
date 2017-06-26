# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

def QuantitiesToMonitor(predictions, labels, dictionary):
    """
    Change which quantities to calculate and monitor while training a model
    For now this is effectively just a placeholder to makes sure NiftyNet's default behaviour stays the same.
    Soon we will have a KL divergence, log likelihood, discriminatory accuracy & other model-specific quantities.
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

    return [quantities_to_monitor, names_of_quantities]
