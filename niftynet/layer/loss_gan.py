# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.engine.application_factory import LossGANFactory
from niftynet.layer.base_layer import Layer


class LossFunction(Layer):
    def __init__(self,
                 loss_type='CrossEntropy',
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
        self._data_loss_func = LossGANFactory.create(type_str)

    def layer_op(self, pred_real, pred_fake, var_scope=None):
        with tf.device('/cpu:0'):
            g_loss = self._data_loss_func['g'](
                pred_fake,
                **self._loss_func_params)
            d_fake = self._data_loss_func['d_fake'](
                pred_fake,
                **self._loss_func_params)
            d_real = self._data_loss_func['d_real'](
                pred_real,
                **self._loss_func_params)
        return g_loss, (d_fake + d_real)


def cross_entropy_function(is_real, softness=.1):
    def cross_entropy_op(pred, **kwargs):
        if is_real:
            target = (1. - softness) * tf.ones_like(pred)
        else:
            target = softness * tf.ones_like(pred)
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,
                                                          labels=target)
        return tf.reduce_mean(entropy)

    return cross_entropy_op


cross_entropy = {'g': cross_entropy_function(True, 0),
                 'd_fake': cross_entropy_function(False, 0),
                 'd_real': cross_entropy_function(True, .1)}
