# -*- coding: utf-8 -*-
"""
Loss functions for regression
"""
from __future__ import absolute_import, print_function, division

import tensorflow as tf

from niftynet.engine.application_factory import LossRegressionFactory
from niftynet.layer.base_layer import Layer


class LossFunction(Layer):
    def __init__(self,
                 loss_type='L2Loss',
                 loss_func_params=None,
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)

        # set loss function and function-specific additional params.
        self._data_loss_func = LossRegressionFactory.create(loss_type)
        self._loss_func_params = \
            loss_func_params if loss_func_params is not None else {}
        self._reshape = True
        if loss_type == 'Cosine':
            self._reshape = False

    def layer_op(self,
                 prediction,
                 ground_truth=None,
                 weight_map=None):
        """
        Compute loss from ``prediction`` and ``ground truth``,
        the computed loss map are weighted by ``weight_map``.

        if ``prediction`` is list of tensors, each element of the list
        will be compared against ``ground_truth` and the weighted by
        ``weight_map``.

        :param prediction: input will be reshaped into
            ``(batch_size, N_voxels, num_classes)``
        :param ground_truth: input will be reshaped into
            ``(batch_size, N_voxels)``
        :param weight_map: input will be reshaped into
            ``(batch_size, N_voxels)``
        :return:
        """

        with tf.device('/cpu:0'):
            batch_size = ground_truth.shape[0].value
            dir_size = 1
            if self._reshape:
                ground_truth = tf.reshape(ground_truth, [batch_size, -1])
                if weight_map is not None:
                    weight_map = tf.reshape(weight_map, [batch_size, -1])
            else:

                dir_size = ground_truth.shape[-1].value
                ground_truth = tf.reshape(ground_truth, [batch_size, -1,
                                                         dir_size])
            if not isinstance(prediction, (list, tuple)):
                prediction = [prediction]

            data_loss = []
            for ind, pred in enumerate(prediction):
                # go through each scale
                def _batch_i_loss(*args):

                    # go through each image in a batch
                    if len(args[0]) == 2:
                        pred_b, ground_truth_b = args[0]
                        weight_map_b = None
                    else:
                        pred_b, ground_truth_b, weight_map_b = args[0]
                    pred_b = tf.reshape(pred_b, tf.shape(ground_truth_b))
                    # pred_b = tf.reshape(pred_b, [-1])
                    # pred_b = tf.Print(tf.cast(pred_b, tf.float32),
                    #                       [tf.shape(
                    #                           pred_b), tf.shape(
                    #                           ground_truth_b)],
                    #                       message='pred_b_shape')


                    loss_params = {
                        'prediction': pred_b,
                        'ground_truth': ground_truth_b,
                        'weight_map': weight_map_b}
                    if self._loss_func_params:
                        loss_params.update(self._loss_func_params)

                    return tf.to_float(self._data_loss_func(**loss_params))

                if weight_map is not None:
                    elements = (pred, ground_truth, weight_map)
                else:
                    elements = (pred, ground_truth)

                loss_batch = tf.map_fn(
                    fn=_batch_i_loss,
                    elems=elements,
                    dtype=tf.float32,
                    parallel_iterations=1)
                data_loss.append(tf.reduce_mean(loss_batch))
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
        residuals = \
            tf.multiply(residuals, weight_map) / tf.reduce_sum(weight_map)
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
        residuals = tf.multiply(residuals, residuals)
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
    that is quadratic for ``|x| <= delta``, and linear for ``|x|> delta``
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


def smooth_l1_loss(prediction, ground_truth, weight_map=None, value_thresh=0.5):
    """
    Similarly to the Huber loss, the residuals are squared below a threshold
    value. In addition they are square above the inverse of this threshold
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :param weight_map:
    :return: mean of the l1 loss across all voxels.
    """
    # Definition of thresholds
    if value_thresh>1:
        value_thresh_max = value_thresh
        value_thresh = 1.0/value_thresh
    else:
        value_thresh_max = 1.0 / value_thresh

    value_correction = value_thresh ** 3 - value_thresh

    value_correction_max = value_thresh_max - value_thresh_max ** 2

    prediction = tf.cast(prediction, dtype=tf.float32)

    ground_truth = tf.cast(ground_truth, dtype=tf.float32)

    absolute_residuals = tf.cast(tf.abs(tf.subtract(prediction,
                                                             ground_truth)),
                                 dtype=tf.float32)

    absolute_residuals = tf.where(absolute_residuals < value_thresh,
                                  value_thresh *
                                           tf.square(absolute_residuals),
                                           absolute_residuals + value_correction)

    absolute_residuals = tf.where(tf.greater(absolute_residuals,value_thresh_max),
                                  tf.square(
        absolute_residuals) + value_correction_max, absolute_residuals)
    if weight_map is not None:

        absolute_residuals = tf.multiply(absolute_residuals, weight_map)
        sum_residuals = tf.reduce_sum(absolute_residuals)

        sum_weights = tf.reduce_sum(weight_map)

    else:
        sum_residuals = tf.reduce_sum(absolute_residuals)
        sum_weights = tf.size(absolute_residuals)
    return tf.truediv(tf.cast(sum_residuals, dtype=tf.float32),
                      tf.cast(sum_weights, dtype=tf.float32))


def cosine_loss(prediction, ground_truth, weight_map=None, to_complete=True):
    '''
    Cosine loss between predicted and ground_truth vectors. The predicted and
     targeted vectors should be unit vectors
    :param prediction:
    :param ground_truth:
    :param weight_map:
    :param to_complete: if the unit vector is to be completed
    :return:
    '''
    if to_complete:
        prediction_complete = tf.reshape(tf.sqrt(1 - tf.minimum(tf.reduce_sum(
            tf.square(
            prediction),-1),1)), [tf.shape(prediction)[0],1])
        ground_truth_complete = tf.reshape(tf.sqrt(1 - tf.minimum(tf.reduce_sum(
            tf.square(
            ground_truth),-1),1)),[tf.shape(prediction)[0],1])

        pred_vect = tf.concat([prediction, prediction_complete], -1)

        gt_vect = tf.concat([ground_truth, ground_truth_complete], -1)
    else:
        pred_vect = prediction
        gt_vect = ground_truth

    if weight_map is None:
        weight_map = tf.ones([tf.shape(prediction)[0]])
    else:
        weight_map = tf.reshape(weight_map, [tf.shape(prediction)[0]])

    pred_vect = pred_vect / tf.maximum(tf.norm(
        pred_vect,ord='euclidean',axis=-1, keep_dims=True), 0.00001)
    gt_vect = gt_vect /tf.maximum(tf.norm(
        gt_vect,ord='euclidean',axis=-1, keep_dims=True), 0.00001)
    loss_init = 1 -tf.reduce_sum(gt_vect * pred_vect, -1)
    weighted_loss = loss_init * weight_map
    loss = tf.reduce_sum(weighted_loss) / tf.reduce_sum(weight_map)

    return loss
