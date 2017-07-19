# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.utilities.misc_common import look_up_operations


class LossFunction(Layer):
    def __init__(self,
                 loss_type='CrossEntropy',
                 loss_func_params={},
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)
        self._loss_func_params = loss_func_params
        self._data_loss_func = None
        self.make_callable_loss_func(loss_type)

    def make_callable_loss_func(self, type_str):
        self._data_loss_func = look_up_operations(type_str, SUPPORTED_OPS)

    def layer_op(self, pred, is_real, var_scope=None):
        with tf.device('/cpu:0'):
            if self._loss_func_params:
                data_loss = self._data_loss_func(pred,
                                                 is_real,
                                                 **self._loss_func_params)
            else:
                data_loss = self._data_loss_func(pred, is_real)
            return data_loss



def cross_entropy(pred, is_real, softness=.1):
    if is_real:
      target = (1.-softness) * tf.ones_like(pred)
    else:
        target = softness * tf.ones_like(pred)
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=target)
    return tf.reduce_mean(entropy)


SUPPORTED_OPS = {"CrossEntropy": cross_entropy,}
