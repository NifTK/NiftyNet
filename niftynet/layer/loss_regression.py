# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import tensorflow as tf

from niftynet.engine.application_factory import LossRegressionFactory
from niftynet.layer.base_layer import Layer


class LossFunction(Layer):
    def __init__(self,
                 loss_type='L2Loss',
                 loss_func_params={},
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)
        self._loss_func_params = loss_func_params
        self._data_loss_func = None
        self.make_callable_loss_func(loss_type)

    def make_callable_loss_func(self, type_str):
        self._data_loss_func = LossRegressionFactory.create(type_str)

    def layer_op(self,
                 prediction,
                 ground_truth=None,
                 weight_map=None,
                 var_scope=None, ):

        with tf.device('/cpu:0'):
            if ground_truth is not None:
                ground_truth = tf.reshape(ground_truth, [-1])

            if not isinstance(prediction, (list, tuple)):
                prediction = [prediction]
            ground_truth = tf.reshape(ground_truth, [-1])
            prediction = tf.reshape(prediction, [-1])
            if weight_map is not None:
                weight_map = tf.reshape(weight_map, [-1])

            data_loss = []

            if self._loss_func_params:
                data_loss.append(self._data_loss_func(
                    prediction, ground_truth, weight_map,
                    **self._loss_func_params))
            else:
                data_loss.append(self._data_loss_func(
                    prediction, ground_truth, weight_map))
            return tf.reduce_mean(data_loss)


def l1_loss(prediction, ground_truth, weight_map=None):
    """
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :return: mean of the l1 loss across all voxels.
    """
    absolute_residuals = tf.abs(tf.subtract(prediction, ground_truth))
    if weight_map is not None:
        absolute_residuals = tf.multiply(absolute_residuals, weight_map)
        sum_residuals = tf.reduce_sum(absolute_residuals)
        sum_weights = tf.reduce_sum(weight_map)
    else:
        sum_residuals = tf.reduce_sum(absolute_residuals)
        sum_weights = tf.size(absolute_residuals)
    return tf.truediv(tf.cast(sum_residuals, dtype=tf.float32),
                      tf.cast(sum_weights, dtype=tf.float32))


def l2_loss(prediction, ground_truth, weight_map=None):
    """
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :return: sum(differences squared) / 2 - Note, no square root
    """

    residuals = tf.subtract(prediction, ground_truth)
    if weight_map is not None:
        residuals = tf.multiply(residuals, weight_map)
    return tf.nn.l2_loss(residuals)


def rmse_loss(prediction, ground_truth, weight_map=None):
    """
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :param weight_map: a weight map for the cost function. .
    :return: sqrt(mean(differences squared))
    """
    if weight_map is not None:
        residuals = tf.subtract(prediction, ground_truth)
        residuals = tf.pow(residuals, 2)
        residuals = tf.multiply(residuals, weight_map)
        return tf.sqrt(tf.reduce_mean(residuals) / tf.reduce_mean(weight_map))
    else:
        return tf.sqrt(tf.losses.mean_squared_error(prediction, ground_truth))


def mae_loss(prediction, ground_truth, weight_map=None):
    """
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :param weight_map: a weight map for the cost function. .
    :return: mean(abs(ground_truth-prediction))
    """
    if weight_map is not None:
        residuals = tf.subtract(prediction, ground_truth)
        residuals = tf.abs(residuals)
        residuals = tf.multiply(residuals, weight_map)
        return tf.reduce_mean(residuals) / tf.reduce_mean(weight_map)
    else:
        return tf.reduce_mean(tf.abs(tf.subtract(prediction, ground_truth)))


def huber_loss(prediction, ground_truth, delta=1.0, weight_map=None):
    """
    The Huber loss is a smooth piecewise loss function
    that is quadratic for |x| <= delta, and linear for |x|> delta
    See https://en.wikipedia.org/wiki/Huber_loss .
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :param delta: the point at which quadratic->linear transition happens.
    :return:
    """
    absolute_residuals = tf.abs(tf.subtract(prediction, ground_truth))
    residual_is_outside_delta = tf.less(delta, absolute_residuals)
    quadratic_residual = 0.5 * absolute_residuals ** 2
    linear_residual = delta * (absolute_residuals - delta / 2)

    voxelwise_loss = tf.where(residual_is_outside_delta,
                              linear_residual,
                              quadratic_residual)
    if weight_map is not None:
        voxelwise_loss = tf.multiply(voxelwise_loss, weight_map)
        sum_weights = tf.reduce_sum(weight_map)
    else:
        sum_weights = tf.to_float(tf.size(absolute_residuals))
    sum_loss = tf.reduce_sum(voxelwise_loss)
    return tf.truediv(sum_loss, sum_weights)


SUPPORTED_OPS = {"L1Loss": l1_loss,
                 "L2Loss": l2_loss,
                 "RMSE": rmse_loss,
                 "Huber": huber_loss}
