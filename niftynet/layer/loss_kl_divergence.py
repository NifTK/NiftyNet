# -*- coding: utf-8 -*-
"""
Loss function for kullback-liebler divergence
"""

import tensorflow as tf

from niftynet.layer.base_layer import Layer


class LossFunction(Layer):

    def __init__(self, name='kl_divergence_loss'):
        super(LossFunction, self).__init__(name=name)

    def layer_op(self, means=None, logvars=None):
        return -0.5 * (1 + logvars - tf.square(means) - tf.exp(logvars))
