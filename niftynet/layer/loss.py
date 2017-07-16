# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.utilities.misc_common import look_up_operations


class LossFunction(Layer):
    def __init__(self,
                 n_class,
                 loss_type='Dice',
                 loss_func_params={},
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)
        self._num_classes = n_class
        self._loss_func_params = loss_func_params
        self._data_loss_func = None
        self.make_callable_loss_func(loss_type)

    def make_callable_loss_func(self, type_str):
        self._data_loss_func = look_up_operations(type_str, SUPPORTED_OPS)

    def layer_op(self, pred, label, weight_map=None, var_scope=None):
        with tf.device('/cpu:0'):
            pred = tf.reshape(pred, [-1, self._num_classes])
            label = tf.reshape(label, [-1])
            if self._loss_func_params:
                data_loss = self._data_loss_func(pred,
                                                 label,
                                                 **self._loss_func_params)
            else:
                data_loss = self._data_loss_func(pred, label)
            return data_loss


# Generalised Dice score with different type weights
def generalised_dice_loss(pred, labels, type_weight='Square'):
    n_voxels = labels.get_shape()[0].value
    n_classes = pred.get_shape()[1].value
    pred = tf.nn.softmax(pred)
    ids = tf.range(n_voxels, dtype=tf.int64)
    ids = tf.stack([ids, labels], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels],dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])

    ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0]) + 0.1
    intersect = tf.sparse_reduce_sum(one_hot * pred, reduction_axes=[0])
    seg_vol = tf.reduce_sum(pred, 0) + 0.1
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\"" \
                         "is not defined.".format(type_weight))

    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    generalised_dice_denominator = \
        tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    return 1 - generalised_dice_score


# Sensitivity Specificity loss function adapted to work for multiple labels
def sensitivity_specificity_loss(pred, labels, r=0.05):
    """
    Function to calculate a multiple-label version of the sensitivity-specificity loss defined in "Deep Convolutional
    Encoder Networks for Multiple Sclerosis Lesion Segmentation", Brosch et al, MICCAI 2015,
    https://link.springer.com/chapter/10.1007/978-3-319-24574-4_1

    error is the sum of r(specificity part) and (1-r)(sensitivity part)

    :param pred: the logits (before softmax).
    :param labels: segmentation labels.
    :param r: the 'sensitivity ratio' (authors suggest values from 0.01-0.10 will have similar effects)
    :return: the loss
    """
    n_voxels = labels.get_shape()[0].value
    n_classes = pred.get_shape()[1].value
    pred = tf.nn.softmax(pred)
    ids =  tf.range(n_voxels, dtype=tf.int64)
    ids = tf.stack([ids, labels], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels],dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])
    one_hot = tf.sparse_tensor_to_dense(one_hot)
    # value of unity everywhere except for the previous 'hot' locations
    one_cold = 1 - one_hot

    # chosen region may contain no voxels of a given label. Prevents nans.
    epsilon_denominator = 1e-5

    squared_error = tf.square(one_hot - pred)
    specificity_part = tf.reduce_sum(squared_error * one_hot, 0) / \
                       (tf.reduce_sum(one_hot, 0) + epsilon_denominator)
    sensitivity_part = (tf.reduce_sum(tf.multiply(squared_error, one_cold), 0) / \
                        (tf.reduce_sum(one_cold, 0) + epsilon_denominator))

    return tf.reduce_sum(r * specificity_part + (1 - r) * sensitivity_part)


def l2_reg_loss(scope):
    if not tf.get_collection('reg_var', scope):
        return 0.0
    return tf.add_n([tf.nn.l2_loss(reg_var) for reg_var in
                     tf.get_collection('reg_var', scope)])


def cross_entropy(pred, labels):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,
                                                             labels=labels)
    return tf.reduce_mean(entropy)


def dice(pred, labels):
    n_voxels = labels.get_shape()[0].value
    n_classes = pred.get_shape()[1].value
    pred = tf.nn.softmax(pred)
    # construct sparse matrix for labels to save space
    ids =  tf.range(n_voxels, dtype=tf.int64)
    ids = tf.stack([ids, labels], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels],dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])
    # dice
    dice_numerator = 2.0 * tf.sparse_reduce_sum(one_hot * pred,
                                                reduction_axes=[0])
    dice_denominator = (tf.reduce_sum(tf.square(pred), reduction_indices=[0]) +
                        tf.sparse_reduce_sum(one_hot, reduction_axes=[0]))
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    dice_score.set_shape([n_classes])
    # minimising (1 - dice_coefficients)
    return 1.0 - tf.reduce_mean(dice_score)


def l1_loss(prediction, ground_truth):
    """
    :param prediction: the current prediction of the ground truth. 
    :param ground_truth: the measurement you are approximating with regression. 
    :return: mean of the l1 loss across all voxels.
    """
    absolute_residuals = tf.abs(tf.subtract(prediction, ground_truth))
    return tf.reduce_mean(absolute_residuals)


def l2_loss(prediction, ground_truth):
    """
    :param prediction: the current prediction of the ground truth. 
    :param ground_truth: the measurement you are approximating with regression. 
    :return: sum(differences squared) / 2 - Note, no square root
    """
    residuals = tf.subtract(prediction, ground_truth)
    return tf.nn.l2_loss(residuals)


def huber_loss(prediction, ground_truth, delta=1.0):
    """
    The Huber loss is a smooth piecewise loss function that is quadratic for |x| <= delta, and linear for |x|> delta
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

    voxelwise_loss = tf.where(residual_is_outside_delta, linear_residual, quadratic_residual)
    return tf.reduce_mean(voxelwise_loss)

SUPPORTED_OPS = {"CrossEntropy": cross_entropy,
                 "Dice": dice,
                 "GDSC": generalised_dice_loss,
                 "SensSpec": sensitivity_specificity_loss,
                 "L1Loss": l1_loss,
                 "L2Loss": l2_loss,
                 "Huber": huber_loss}
