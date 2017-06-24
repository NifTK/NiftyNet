# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

# Functions for changing which quantities we monitor while training a model

def QuantitiesToMonitor(predictions, labels, param):
    """
    For now this is effectively just a placeholder to makes sure NiftyNet's default behaviour stays the same.
    Soon we will have KL divergence, log likelihood, discriminatory accuracy & other model-specific training summaries
    that we can switch on and off from the configuration file.
    :param predictions: outputs of the model.
    :param labels: ground truth.
    :param param: the imported configuration file.
    :return: a list of quantities to monitor during training in 'quantities_to_monitor', with their human-readable
    names in 'names_of_quantities'
    """
    quantities_to_monitor = []
    names_of_quantities = []

    if hasattr(param, 'monitor_miss_rate'):
        if param.monitor_miss_rate=='True':
            if param.monitor_miss_rate:
                quantities_to_monitor.append(tf.reduce_mean(tf.cast(
                    tf.not_equal(tf.argmax(predictions, -1), labels[..., 0]),
                    dtype=tf.float32)))
                names_of_quantities.append('miss rate')
    else:
        quantities_to_monitor.append(tf.reduce_mean(tf.cast(
            tf.not_equal(tf.argmax(predictions, -1), labels[..., 0]),
            dtype=tf.float32)))
        names_of_quantities.append('miss rate')

    return [quantities_to_monitor, names_of_quantities]
