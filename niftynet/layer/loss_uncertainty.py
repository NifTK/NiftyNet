# -*- coding: utf-8 -*-
"""
Loss functions for networks that incorporate aleatoric uncertainty
"""


import numpy as np
import tensorflow as tf
from niftynet.layer.loss_segmentation import dice_plus_xent_loss

from niftynet.layer.base_layer import Layer

def _dicexent_loss(prediction, ground_truth, weight_map=None):
    return dice_plus_xent_loss(prediction, ground_truth, weight_map)


def _l2_loss(prediction, ground_truth, weight_map=None):
    return tf.square(prediction - ground_truth)


class LossFunction(Layer):

    _log2_pi_by_2 = np.log(2 * np.pi) / 2
    _loss_functions = {
        'dice_plus_xent': _dicexent_loss, 'l2': _l2_loss
    }


    def __init__(self, loss_function_str, name='uncertainty_loss_function'):
        super(LossFunction, self).__init__(name=name)
        self._loss_function =\
            LossFunction._loss_functions.get(loss_function_str, None)
        if self._loss_function is None:
            raise ValueError("Parameter 'loss_function_str' doesn't match an"
                             "available loss function")

    def layer_op(self,
            prediction_means=None, prediction_logvars=None,
            ground_truth=None, weight_map=None):
        with tf.device('/cpu:0'):
            error_term = self._loss_function(prediction_means, ground_truth)
            if prediction_logvars is None:
                return tf.reduce_mean(error_term)
            else:
                loss = -LossFunction._log2_pi_by_2 -\
                       prediction_logvars -\
                       0.5 * error_term * tf.exp(2 * -prediction_logvars)

                return -tf.reduce_mean(loss)


