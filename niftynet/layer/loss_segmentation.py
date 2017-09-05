# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.engine.application_factory import LossSegmentationFactory

M_tree = np.array([[0., 1., 1., 1., 1.],
                   [1., 0., 0.6, 0.2, 0.5],
                   [1., 0.6, 0., 0.6, 0.7],
                   [1., 0.2, 0.6, 0., 0.5],
                   [1., 0.5, 0.7, 0.5, 0.]], dtype=np.float64)


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
        self._data_loss_func = LossSegmentationFactory.create(type_str)

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
            # prediction should be a list for holistic networks
            if self._num_classes > 0:
                # reshape the prediction to [n_voxels , num_classes]
                prediction = [tf.reshape(pred, [-1, self._num_classes])
                              for pred in prediction]
            if weight_map is not None:
                weight_map = tf.reshape(weight_map, [-1])

            data_loss = []
            for pred in prediction:
                if self._loss_func_params:
                    data_loss.append(self._data_loss_func(
                        pred, ground_truth, weight_map,
                        **self._loss_func_params))
                else:
                    data_loss.append(self._data_loss_func(
                        pred, ground_truth, weight_map))
            return tf.reduce_mean(data_loss)


def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map,
                          type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in Sudre, C. et. al.
     (2017) Generalised Dice overlap as a deep learning loss function for highly
      unbalanced segmentations. DLMIA 2017
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
    between Square (square of inverse of volume), Simple (inverse of volume)
    and Uniform (no weighting))
    :return: the loss
    """
    n_voxels = ground_truth.get_shape()[0].value
    n_classes = prediction.get_shape()[1].value
    prediction = tf.nn.softmax(prediction)
    weight_map_nclasses = tf.reshape(
        tf.tile(weight_map, [n_classes]), prediction.get_shape())

    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels], dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])

    ref_vol = tf.sparse_reduce_sum(
        weight_map_nclasses * one_hot, reduction_axes=[0]) + 0.1
    intersect = tf.sparse_reduce_sum(
        weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
    seg_vol = tf.reduce_sum(
        tf.multiply(weight_map_nclasses, prediction), 0) + 0.1
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


def sensitivity_specificity_loss(prediction,
                                 ground_truth,
                                 weight_map=None,
                                 r=0.05):
    """
    Function to calculate a multiple-ground_truth version of
    the sensitivity-specificity loss defined in "Deep Convolutional
    Encoder Networks for Multiple Sclerosis Lesion Segmentation",
    Brosch et al, MICCAI 2015,
    https://link.springer.com/chapter/10.1007/978-3-319-24574-4_1

    error is the sum of r(specificity part) and (1-r)(sensitivity part)

    :param pred: the logits (before softmax).
    :param ground_truth: segmentation ground_truth.
    :param r: the 'sensitivity ratio'
        (authors suggest values from 0.01-0.10 will have similar effects)
    :return: the loss
    """

    n_voxels = ground_truth.get_shape()[0].value
    n_classes = prediction.get_shape()[1].value
    prediction = tf.nn.softmax(prediction)
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)

    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels], dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])
    one_hot = tf.sparse_tensor_to_dense(one_hot)
    # value of unity everywhere except for the previous 'hot' locations
    one_cold = 1 - one_hot

    # chosen region may contain no voxels of a given label. Prevents nans.
    epsilon_denominator = 1e-5

    squared_error = tf.square(one_hot - prediction)
    specificity_part = tf.reduce_sum(
        squared_error * one_hot, 0) / \
                       (tf.reduce_sum(one_hot, 0) + epsilon_denominator)
    sensitivity_part = (tf.reduce_sum(tf.multiply(squared_error, one_cold), 0) / \
                        (tf.reduce_sum(one_cold, 0) + epsilon_denominator))

    return tf.reduce_sum(r * specificity_part + (1 - r) * sensitivity_part)


def l2_reg_loss(scope):
    if not tf.get_collection('reg_var', scope):
        return 0.0
    return tf.add_n([tf.nn.l2_loss(reg_var) for reg_var in
                     tf.get_collection('reg_var', scope)])


def cross_entropy(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the cross-entropy loss function
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :return: the cross-entropy loss
    """
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=ground_truth)
    if weight_map is not None:
        weight_map = tf.size(entropy) / tf.reduce_sum(weight_map) * \
                     weight_map
        entropy = tf.multiply(entropy, weight_map)
    return tf.reduce_mean(entropy)


def wasserstein_disagreement_map(prediction, ground_truth, M):
    """
    Function to calculate the pixel-wise Wasserstein distance between the
    flattened pred_proba and the flattened labels (ground_truth) with respect
    to the distance matrix on the label space M.

    :param prediction: the logits after softmax
    :param ground_truth: segmentation ground_truth
    :param M: distance matrix on the label space
    :return: the pixelwise distance map (wass_dis_map)
    """
    # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
    # wrt the distance matrix on the label space M
    n_classes = prediction.get_shape()[1].value
    unstack_labels = tf.unstack(ground_truth, axis=-1)
    unstack_labels = tf.cast(unstack_labels, dtype=tf.float64)
    unstack_pred = tf.unstack(prediction, axis=-1)
    unstack_pred = tf.cast(unstack_pred, dtype=tf.float64)
    # print("shape of M", M.shape, "unstacked labels", unstack_labels,
    #       "unstacked pred" ,unstack_pred)
    # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
    pairwise_correlations = []
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise_correlations.append(
                M[i, j] * tf.multiply(unstack_pred[i], unstack_labels[j]))
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map


def generalised_wasserstein_dice_loss(prediction,
                                      ground_truth,
                                      weight_map=None):
    """
    Function to calculate the Generalised Wasserstein Dice Loss defined in
    Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score for Imbalanced
    Multi-class Segmentation using Holistic Convolutional Networks.
    MICCAI 2017 (BrainLes)
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    # apply softmax to pred scores
    ground_truth = tf.cast(ground_truth, dtype=tf.int64)
    pred_proba = tf.nn.softmax(tf.cast(prediction, dtype=tf.float64))
    n_classes = prediction.get_shape()[1].value
    n_voxels = prediction.get_shape()[0].value
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)

    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels], dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])
    one_hot = tf.sparse_tensor_to_dense(one_hot)
    # M = tf.cast(M, dtype=tf.float64)
    # compute disagreement map (delta)
    M = M_tree
    # print("M shape is ", M.shape, pred_proba, one_hot)
    delta = wasserstein_disagreement_map(pred_proba, one_hot, M)
    # compute generalisation of all error for multi-class seg
    all_error = tf.reduce_sum(delta)
    # compute generalisation of true positives for multi-class seg
    one_hot = tf.cast(one_hot, dtype=tf.float64)
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot),
        axis=1)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)
    return tf.cast(WGDL,dtype=tf.float32)


def dice_nosquare(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the classical dice loss
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    n_voxels = ground_truth.get_shape()[0].value
    n_classes = prediction.get_shape()[1].value
    prediction = tf.nn.softmax(prediction)
    weight_map_nclasses = tf.reshape(
        tf.tile(weight_map, [n_classes]), prediction.get_shape())
    # construct sparse matrix for ground_truth to save space
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels], dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])
    # dice
    dice_numerator = 2.0 * tf.sparse_reduce_sum(
        weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
    dice_denominator = (
        tf.reduce_sum(prediction, reduction_indices=[0]) +
        tf.sparse_reduce_sum(one_hot, reduction_axes=[0]))
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    dice_score.set_shape([n_classes])
    # minimising (1 - dice_coefficients)
    return 1.0 - tf.reduce_mean(dice_score)


def dice(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the dice loss with the definition given in Milletari,
     F., Navab, N., & Ahmadi, S. A. (2016) V-net: Fully convolutional neural
     networks for volumetric medical image segmentation. 3DV 2016 using a
     square in the denominator
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    ground_truth = tf.to_int64(ground_truth)
    prediction = tf.cast(prediction, tf.float32)
    prediction = tf.nn.softmax(prediction)
    ids = tf.range(tf.to_int64(tf.shape(ground_truth)[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(tf.shape(prediction)))
    # if weight_map is not None:
    #    weight_map_nclasses = tf.reshape(
    #        tf.tile(weight_map, [n_classes]), prediction.get_shape())
    #    dice_numerator = 2.0 * tf.sparse_reduce_sum(
    #        weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
    # else:
    dice_numerator = 2.0 * tf.sparse_reduce_sum(
        one_hot * prediction, reduction_axes=[0])
    dice_denominator = \
        tf.reduce_sum(tf.square(prediction), reduction_indices=[0]) + \
        tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    # dice_score.set_shape([n_classes])
    # minimising (1 - dice_coefficients)
    return 1.0 - tf.reduce_mean(dice_score)


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


def huber_loss(prediction, ground_truth, delta=1.0, weight_map=None):
    """
    The Huber loss is a smooth piecewise loss function
    that is quadratic for |x| <= delta, and linear for |x|> delta
    See https://en.wikipedia.org/wiki/Huber_loss .
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :param delta: the point at which quadratic->linear transition happens.
    :return: the loss
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


SUPPORTED_OPS = {"CrossEntropy": cross_entropy,
                 "Dice": dice,
                 "Dice_NS": dice_nosquare,
                 "GDSC": generalised_dice_loss,
                 "WGDL": generalised_wasserstein_dice_loss,
                 "SensSpec": sensitivity_specificity_loss,
                 "L1Loss": l1_loss,
                 "L2Loss": l2_loss,
                 "Huber": huber_loss}
