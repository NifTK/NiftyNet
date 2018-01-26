# -*- coding: utf-8 -*-
"""
Loss functions for multi-class segmentation
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.engine.application_factory import LossSegmentationFactory
from niftynet.layer.base_layer import Layer

M_tree = np.array([[0., 1., 1., 1., 1.],
                   [1., 0., 0.6, 0.2, 0.5],
                   [1., 0.6, 0., 0.6, 0.7],
                   [1., 0.2, 0.6, 0., 0.5],
                   [1., 0.5, 0.7, 0.5, 0.]], dtype=np.float64)


class LossFunction(Layer):
    def __init__(self,
                 n_class,
                 loss_type='Dice',
                 softmax=True,
                 loss_func_params=None,
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)
        assert n_class > 0, \
            "Number of classes for segmentation loss should be positive."
        self._num_classes = n_class

        self._softmax = bool(softmax)
        # set loss function and function-specific additional params.
        self._data_loss_func = LossSegmentationFactory.create(loss_type)
        self._loss_func_params = \
            loss_func_params if loss_func_params is not None else dict()

        if self._data_loss_func.__name__ == 'cross_entropy':
            tf.logging.info(
                'Cross entropy loss function calls '
                'tf.nn.sparse_softmax_cross_entropy_with_logits '
                'which always performs a softmax internally.')
            self._softmax = False

    def layer_op(self,
                 prediction,
                 ground_truth,
                 weight_map=None):
        """
        Compute loss from `prediction` and `ground truth`,
        the computed loss map are weighted by `weight_map`.

        if `prediction `is list of tensors, each element of the list
        will be compared against `ground_truth` and the weighted by
        `weight_map`. (Assuming the same gt and weight across scales)

        :param prediction: input will be reshaped into
            ``(batch_size, N_voxels, num_classes)``
        :param ground_truth: input will be reshaped into
            ``(batch_size, N_voxels, ...)``
        :param weight_map: input will be reshaped into
            ``(batch_size, N_voxels, ...)``
        :return:
        """

        with tf.device('/cpu:0'):

            # prediction should be a list for multi-scale losses
            # single scale ``prediction`` is converted to ``[prediction]``
            if not isinstance(prediction, (list, tuple)):
                prediction = [prediction]

            data_loss = []
            for ind, pred in enumerate(prediction):
                # go through each scale

                loss_batch = []
                for b_ind, pred_b in enumerate(tf.unstack(pred, axis=0)):
                    # go through each image in a batch

                    pred_b = tf.reshape(pred_b, [-1, self._num_classes])
                    # performs softmax if required
                    if self._softmax:
                        pred_b = tf.cast(pred_b, dtype=tf.float32)
                        pred_b = tf.nn.softmax(pred_b)

                    # reshape pred, ground_truth, weight_map to the same
                    # size: (n_voxels, num_classes)
                    # if the ground_truth has only one channel, the shape
                    # becomes: (n_voxels,)
                    spatial_shape = pred_b.shape.as_list()[:-1]
                    ref_shape = spatial_shape + [-1]
                    ground_truth_b = tf.reshape(ground_truth[b_ind], ref_shape)
                    if ground_truth_b.shape.as_list()[-1] == 1:
                        ground_truth_b = tf.squeeze(ground_truth_b, axis=-1)
                    if weight_map is not None:
                        weight_b = tf.reshape(weight_map[b_ind], ref_shape)
                        if weight_b.shape.as_list()[-1] == 1:
                            weight_b = tf.squeeze(weight_b, axis=-1)
                    else:
                        weight_b = None

                    # preparing loss function parameters
                    loss_params = {
                        'prediction': pred_b,
                        'ground_truth': ground_truth_b,
                        'weight_map': weight_b}
                    if self._loss_func_params:
                        loss_params.update(self._loss_func_params)

                    # loss for each batch over spatial dimensions
                    loss_batch.append(self._data_loss_func(**loss_params))
                # loss averaged over batch
                data_loss.append(tf.reduce_mean(loss_batch))
            # loss averaged over multiple scales
            return tf.reduce_mean(data_loss)


def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017

    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    ground_truth = tf.to_int64(ground_truth)
    n_voxels = ground_truth.shape[0].value
    n_classes = prediction.shape[1].value
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels], dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])

    if weight_map is not None:
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
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

    :param prediction: the logits
    :param ground_truth: segmentation ground_truth.
    :param r: the 'sensitivity ratio'
        (authors suggest values from 0.01-0.10 will have similar effects)
    :return: the loss
    """
    if weight_map is not None:
        # raise NotImplementedError
        tf.logging.warning('Weight map specified but not used.')

    ground_truth = tf.to_int64(ground_truth)
    n_voxels = ground_truth.shape[0].value
    n_classes = prediction.shape[1].value
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
    sensitivity_part = \
        (tf.reduce_sum(tf.multiply(squared_error, one_cold), 0) /
         (tf.reduce_sum(one_cold, 0) + epsilon_denominator))

    return tf.reduce_sum(r * specificity_part + (1 - r) * sensitivity_part)


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
        weight_map = tf.cast(tf.size(entropy), dtype=tf.float32) / \
                     tf.reduce_sum(weight_map) * weight_map
        entropy = tf.multiply(entropy, weight_map)
    return tf.reduce_mean(entropy)


def wasserstein_disagreement_map(
        prediction, ground_truth, weight_map=None, M=None):
    """
    Function to calculate the pixel-wise Wasserstein distance between the
    flattened prediction and the flattened labels (ground_truth) with respect
    to the distance matrix on the label space M.

    :param prediction: the logits after softmax
    :param ground_truth: segmentation ground_truth
    :param M: distance matrix on the label space
    :return: the pixelwise distance map (wass_dis_map)
    """
    if weight_map is not None:
        # raise NotImplementedError
        tf.logging.warning('Weight map specified but not used.')

    assert M is not None, "Distance matrix is required."
    # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
    # wrt the distance matrix on the label space M
    n_classes = prediction.shape[1].value
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

        Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score
        for Imbalanced Multi-class Segmentation using Holistic
        Convolutional Networks.MICCAI 2017 (BrainLes)

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    if weight_map is not None:
        # raise NotImplementedError
        tf.logging.warning('Weight map specified but not used.')

    # apply softmax to pred scores
    ground_truth = tf.cast(ground_truth, dtype=tf.int64)
    n_classes = prediction.shape[1].value
    n_voxels = prediction.shape[0].value
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)

    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels], dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])
    one_hot = tf.sparse_tensor_to_dense(one_hot)
    # M = tf.cast(M, dtype=tf.float64)
    # compute disagreement map (delta)
    M = M_tree
    delta = wasserstein_disagreement_map(prediction, one_hot, M=M)
    # compute generalisation of all error for multi-class seg
    all_error = tf.reduce_sum(delta)
    # compute generalisation of true positives for multi-class seg
    one_hot = tf.cast(one_hot, dtype=tf.float64)
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot),
        axis=1)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)
    return tf.cast(WGDL, dtype=tf.float32)


def dice_nosquare(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the classical dice loss

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    ground_truth = tf.to_int64(ground_truth)
    n_voxels = ground_truth.shape[0].value
    n_classes = prediction.shape[1].value
    # construct sparse matrix for ground_truth to save space
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=tf.ones([n_voxels], dtype=tf.float32),
                              dense_shape=[n_voxels, n_classes])
    # dice
    if weight_map is not None:
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(prediction * weight_map_nclasses,
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(weight_map_nclasses * one_hot,
                                 reduction_axes=[0])
    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(one_hot * prediction,
                                                    reduction_axes=[0])
        dice_denominator = tf.reduce_sum(prediction, reduction_indices=[0]) + \
                           tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    # dice_score.set_shape([n_classes])
    # minimising (1 - dice_coefficients)
    return 1.0 - tf.reduce_mean(dice_score)


def dice(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the dice loss with the definition given in

        Milletari, F., Navab, N., & Ahmadi, S. A. (2016)
        V-net: Fully convolutional neural
        networks for volumetric medical image segmentation. 3DV 2016

    using a square in the denominator

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    ground_truth = tf.to_int64(ground_truth)
    prediction = tf.cast(prediction, tf.float32)
    ids = tf.range(tf.to_int64(tf.shape(ground_truth)[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(tf.shape(prediction)))
    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.reshape(
            tf.tile(weight_map, [n_classes]), prediction.get_shape())
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * tf.square(prediction),
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
                                 reduction_axes=[0])
    else:
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


def dice_dense(prediction, ground_truth, weight_map=None):
    """
    Computing mean-class Dice similarity.
    This function assumes one-hot encoded ground truth

    :param prediction: last dimension should have ``num_classes``
    :param ground_truth: segmentation ground truth (encoded as a binary matrix)
        last dimension should be ``num_classes``
    :param weight_map:
    :return: ``1.0 - mean(Dice similarity per class)``
    """

    if weight_map is not None:
        raise NotImplementedError
    prediction = tf.cast(prediction, dtype=tf.float32)
    ground_truth = tf.cast(ground_truth, dtype=tf.float32)
    ground_truth = tf.reshape(ground_truth, prediction.shape)
    # computing Dice over the spatial dimensions
    reduce_axes = list(range(len(prediction.shape) - 1))
    dice_numerator = 2.0 * tf.reduce_sum(
        prediction * ground_truth, axis=reduce_axes)
    dice_denominator = \
        tf.reduce_sum(prediction, axis=reduce_axes) + \
        tf.reduce_sum(ground_truth, axis=reduce_axes)
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    return 1.0 - tf.reduce_mean(dice_score)
