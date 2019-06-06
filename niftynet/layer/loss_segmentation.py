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

        data_loss_function_name = self._data_loss_func.__name__
        if data_loss_function_name.startswith('cross_entropy') \
                or 'xent' in data_loss_function_name:
            tf.logging.info(
                'Cross entropy loss function calls '
                'tf.nn.sparse_softmax_cross_entropy_with_logits '
                'which always performs a softmax internally.')
            self._softmax = False

    def layer_op(self, prediction, ground_truth, weight_map=None):
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
                def _batch_i_loss(*args):
                    """
                    loss for the `b_id`-th batch (over spatial dimensions)

                    :param b_id:
                    :return:
                    """
                    # unpacking input from map_fn elements
                    if len(args[0]) == 2:
                        # pred and ground_truth
                        pred_b, ground_truth_b = args[0]
                        weight_b = None
                    else:
                        pred_b, ground_truth_b, weight_b = args[0]

                    pred_b = tf.reshape(pred_b, [-1, self._num_classes])
                    # performs softmax if required
                    if self._softmax:
                        pred_b = tf.cast(pred_b, dtype=tf.float32)
                        pred_b = tf.nn.softmax(pred_b)

                    # reshape pred, ground_truth, weight_map to the same
                    # size: (n_voxels, num_classes)
                    # if the ground_truth has only one channel, the shape
                    # becomes: (n_voxels,)
                    if not pred_b.shape.is_fully_defined():
                        ref_shape = tf.stack(
                            [tf.shape(pred_b)[0], tf.constant(-1)], 0)
                    else:
                        ref_shape = pred_b.shape.as_list()[:-1] + [-1]

                    ground_truth_b = tf.reshape(ground_truth_b, ref_shape)
                    if ground_truth_b.shape.as_list()[-1] == 1:
                        ground_truth_b = tf.squeeze(ground_truth_b, axis=-1)

                    if weight_b is not None:
                        weight_b = tf.reshape(weight_b, ref_shape)
                        if weight_b.shape.as_list()[-1] == 1:
                            weight_b = tf.squeeze(weight_b, axis=-1)

                    # preparing loss function parameters
                    loss_params = {
                        'prediction': pred_b,
                        'ground_truth': ground_truth_b,
                        'weight_map': weight_b}
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

                # loss averaged over batch
                data_loss.append(tf.reduce_mean(loss_batch))
            # loss averaged over multiple scales
            return tf.reduce_mean(data_loss)


def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.

    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot


def undecided_loss(prediction, ground_truth, weight_map=None):
    """

    :param prediction:
    :param ground_truth:
    :param weight_map:
    :return:
    """
    ratio_undecided = 1.0/tf.cast(tf.shape(prediction)[-1], tf.float32)
    res_undecided = tf.reciprocal(tf.reduce_mean(tf.abs(prediction -
                                                 ratio_undecided), -1) + 0.0001)
    if weight_map is None:
        return tf.reduce_mean(res_undecided)
    else:
        res_undecided = tf.Print(tf.cast(res_undecided, tf.float32), [tf.shape(
            res_undecided), tf.shape(weight_map), tf.shape(
                res_undecided*weight_map)], message='test_printshape_und')
        return tf.reduce_sum(res_undecided * weight_map /
                             tf.reduce_sum(weight_map))


def volume_enforcement(prediction, ground_truth, weight_map=None, eps=0.001,
                       hard=False):
    """
    Computing a volume enforcement loss to ensure that the obtained volumes are
    close and avoid empty results when something is expected
    :param prediction:
    :param ground_truth: labels
    :param weight_map: potential weight map to apply
    :param eps: epsilon to use as regulariser
    :return:
    """

    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    gt_red = tf.sparse_reduce_sum(one_hot, 0)
    pred_red = tf.reduce_sum(prediction, 0)
    if hard:
        pred_red  = tf.sparse_reduce_sum(labels_to_one_hot(tf.argmax(
            prediction,-1),tf.shape(prediction)[-1]), 0)

    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.tile(tf.expand_dims(tf.reshape(weight_map,
                                                                [-1]), 1),
                                      [1, n_classes])
        gt_red = tf.sparse_reduce_sum(weight_map_nclasses * one_hot,
                                      reduction_axes=[0])
        pred_red = tf.reduce_sum(weight_map_nclasses * prediction, 0)

    return tf.reduce_mean(tf.sqrt(tf.square((gt_red+eps)/(pred_red+eps) -
                                            (pred_red+eps)/(gt_red+eps))))


def volume_enforcement_fin(prediction, ground_truth, weight_map=None,
                           eps=0.001):
    """
    Computing a volume enforcement loss to ensure that the obtained volumes are
     close and avoid empty results when something is expected
    :param prediction:
    :param ground_truth:
    :param weight_map:
    :param eps:
    :return:
    """

    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
    gt_red = tf.sparse_reduce_sum(one_hot, 0)
    pred_red = tf.sparse_reduce_sum(labels_to_one_hot(tf.argmax(
            prediction,-1),tf.shape(prediction)[-1]), 0)

    if weight_map is not None:
        n_classes = prediction.shape[1].value
        weight_map_nclasses = tf.tile(tf.expand_dims(tf.reshape(weight_map,
                                                                [-1]), 1),
                                      [1, n_classes])
        gt_red = tf.sparse_reduce_sum(weight_map_nclasses * one_hot,
                                      reduction_axes=[0])
        pred_red = tf.sparse_reduce_sum(labels_to_one_hot(tf.argmax(
            prediction, -1), tf.shape(prediction)[-1]) * weight_map_nclasses, 0)

    return tf.reduce_mean(tf.sqrt(tf.square((gt_red+eps)/(pred_red+eps)
                                            - (pred_red+eps)/(gt_red+eps))))



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
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        # weight_map_nclasses = tf.reshape(
        #     tf.tile(weight_map, [num_classes]), prediction.get_shape())
        weight_map_nclasses = tf.tile(
            tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, num_classes])
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
    # generalised_dice_denominator = \
    #     tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    return 1 - generalised_dice_score


def dice_plus_xent_loss(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the loss used in https://arxiv.org/pdf/1809.10486.pdf,
    no-new net, Isenseee et al (used to win the Medical Imaging Decathlon).

    It is the sum of the cross-entropy and the Dice-loss.

    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :return: the loss (cross_entropy + Dice)

    """
    num_classes = tf.shape(prediction)[-1]

    prediction = tf.cast(prediction, tf.float32)
    loss_xent = cross_entropy(prediction, ground_truth, weight_map=weight_map)

    # Dice as according to the paper:
    one_hot = labels_to_one_hot(ground_truth, num_classes=num_classes)
    softmax_of_logits = tf.nn.softmax(prediction)

    if weight_map is not None:
        weight_map_nclasses = tf.tile(
            tf.reshape(weight_map, [-1, 1]), [1, num_classes])
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * softmax_of_logits,
            reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(weight_map_nclasses * softmax_of_logits,
                          reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot * weight_map_nclasses,
                                 reduction_axes=[0])
    else:
        dice_numerator = 2.0 * tf.sparse_reduce_sum(
            one_hot * softmax_of_logits, reduction_axes=[0])
        dice_denominator = \
            tf.reduce_sum(softmax_of_logits, reduction_indices=[0]) + \
            tf.sparse_reduce_sum(one_hot, reduction_axes=[0])

    epsilon = 0.00001
    loss_dice = -(dice_numerator + epsilon) / (dice_denominator + epsilon)
    dice_numerator = tf.Print(
        dice_denominator, [dice_numerator, dice_denominator, loss_dice])

    return loss_dice + loss_xent


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

    prediction = tf.cast(prediction, tf.float32)
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    one_hot = tf.sparse_tensor_to_dense(one_hot)
    # value of unity everywhere except for the previous 'hot' locations
    one_cold = 1 - one_hot

    # chosen region may contain no voxels of a given label. Prevents nans.
    epsilon = 1e-5

    squared_error = tf.square(one_hot - prediction)
    specificity_part = tf.reduce_sum(
        squared_error * one_hot, 0) / \
                       (tf.reduce_sum(one_hot, 0) + epsilon)
    sensitivity_part = \
        (tf.reduce_sum(tf.multiply(squared_error, one_cold), 0) /
         (tf.reduce_sum(one_cold, 0) + epsilon))

    return tf.reduce_sum(r * specificity_part + (1 - r) * sensitivity_part)


def cross_entropy(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the cross-entropy loss function

    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :return: the cross-entropy loss
    """
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]

    # TODO trace this back:
    ground_truth = tf.cast(ground_truth, tf.int32)

    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=ground_truth)

    if weight_map is None:
        return tf.reduce_mean(entropy)

    weight_sum = tf.maximum(tf.reduce_sum(weight_map), 1e-6)
    return tf.reduce_sum(entropy * weight_map / weight_sum)


def cross_entropy_dense(prediction, ground_truth, weight_map=None):
    if weight_map is not None:
        raise NotImplementedError

    entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=ground_truth)
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
    num_classes = prediction.shape[1].value
    ground_truth.set_shape(prediction.shape)
    unstack_labels = tf.unstack(ground_truth, axis=-1)
    unstack_labels = tf.cast(unstack_labels, dtype=tf.float64)
    unstack_pred = tf.unstack(prediction, axis=-1)
    unstack_pred = tf.cast(unstack_pred, dtype=tf.float64)
    # print("shape of M", M.shape, "unstacked labels", unstack_labels,
    #       "unstacked pred" ,unstack_pred)
    # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
    pairwise_correlations = []
    for i in range(num_classes):
        for j in range(num_classes):
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

    prediction = tf.cast(prediction, tf.float32)
    num_classes = prediction.shape[1].value
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

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
        tf.multiply(tf.constant(M[0, :num_classes], dtype=tf.float64), one_hot),
        axis=1)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)
    return tf.cast(WGDL, dtype=tf.float32)


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
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        weight_map_nclasses = tf.tile(tf.expand_dims(
            tf.reshape(weight_map, [-1]), 1), [1, num_classes])
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
    epsilon = 0.00001

    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    # dice_score.set_shape([num_classes])
    # minimising (1 - dice_coefficients)
    return 1.0 - tf.reduce_mean(dice_score)


def dice_nosquare(prediction, ground_truth, weight_map=None):
    """
    Function to calculate the classical dice loss

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    # dice
    if weight_map is not None:
        num_classes = prediction.shape[1].value
        weight_map_nclasses = tf.tile(tf.expand_dims(
            tf.reshape(weight_map, [-1]), 1), [1, num_classes])
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
    epsilon = 0.00001

    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    # dice_score.set_shape([num_classes])
    # minimising (1 - dice_coefficients)
    return 1.0 - tf.reduce_mean(dice_score)


def tversky(prediction, ground_truth, weight_map=None, alpha=0.5, beta=0.5):
    """
    Function to calculate the Tversky loss for imbalanced data

        Sadegh et al. (2017)

        Tversky loss function for image segmentation
        using 3D fully convolutional deep networks

    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param weight_map:
    :return: the loss
    """
    prediction = tf.to_float(prediction)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
    one_hot = tf.sparse_tensor_to_dense(one_hot)

    p0 = prediction
    p1 = 1 - prediction
    g0 = one_hot
    g1 = 1 - one_hot

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        weight_map_flattened = tf.reshape(weight_map, [-1])
        weight_map_expanded = tf.expand_dims(weight_map_flattened, 1)
        weight_map_nclasses = tf.tile(weight_map_expanded, [1, num_classes])
    else:
        weight_map_nclasses = 1

    tp = tf.reduce_sum(weight_map_nclasses * p0 * g0)
    fp = alpha * tf.reduce_sum(weight_map_nclasses * p0 * g1)
    fn = beta * tf.reduce_sum(weight_map_nclasses * p1 * g0)

    EPSILON = 0.00001
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    return 1.0 - tf.reduce_mean(score)


def dice_dense(prediction, ground_truth, weight_map=None):
    """
    Computing mean-class Dice similarity.

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
        tf.reduce_sum(tf.square(prediction), axis=reduce_axes) + \
        tf.reduce_sum(tf.square(ground_truth), axis=reduce_axes)

    epsilon = 0.00001

    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    return 1.0 - tf.reduce_mean(dice_score)


def dice_dense_nosquare(prediction, ground_truth, weight_map=None):
    """
    Computing mean-class Dice similarity with no square terms in the denominator

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
    epsilon = 0.00001

    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    return 1.0 - tf.reduce_mean(dice_score)
