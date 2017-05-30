# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from layer.base import Layer
from utilities.misc_common import damerau_levenshtein_distance


class LossFunction(Layer):
    def __init__(self,
                 n_class,
                 loss_type='Dice',
                 reg_type='L2',
                 decay=0.0,
                 loss_func_params={},
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)
        self._num_classes = 0
        self._data_loss_func = None
        self._regularisation_loss_func = None
        self._decay = None
        self._loss_func_params = None
        self._set_all_fields(
            n_class, loss_type, reg_type, decay, loss_func_params)
        print('Training loss: {}_loss + ({})*{}_loss'.format(
            loss_type, decay, reg_type))

    def _set_all_fields(self,
                        n_class,
                        loss_type,
                        reg_type,
                        decay,
                        loss_func_params):
        self.num_classes = n_class
        self.data_loss_func = loss_type
        self.regularisation_loss_func = reg_type
        self.decay = decay
        self.loss_func_params = loss_func_params

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data_loss_func(self):
        return self._data_loss_func

    @property
    def regularisation_loss_func(self):
        return self._regularisation_loss_func

    @property
    def decay(self):
        return self._decay

    @property
    def loss_func_params(self):
        return self._loss_func_params

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value

    @data_loss_func.setter
    def data_loss_func(self, type_str):
        if type_str in SUPPORTED_OPS.keys():
            self._data_loss_func = SUPPORTED_OPS[type_str]
        else:
            edit_distances = {}
            for loss_name in SUPPORTED_OPS.keys():
                edit_distance = damerau_levenshtein_distance(loss_name,
                                                             type_str)
                if edit_distance <= 3:
                    edit_distances[loss_name] = edit_distance
            if edit_distances:
                guess_at_correct_spelling = min(edit_distances,
                                                key=edit_distances.get)
                raise ValueError('By "{0}", did you mean "{1}"?\n '
                                 '"{0}" is not a valid loss.'.format(
                    type_str, guess_at_correct_spelling))
            else:
                raise ValueError("Loss type \"{}\" "
                                 "is not found.".format(type_str))

    @regularisation_loss_func.setter
    def regularisation_loss_func(self, type_str):
        if type_str.upper() == "L2":
            self._regularisation_loss_func = l2_reg_loss
        else:
            raise NotImplementedError(
                'regularisation parameters not implemented')

    @decay.setter
    def decay(self, value):
        self._decay = value

    @loss_func_params.setter
    def loss_func_params(self, value):
        self._loss_func_params = value

    def layer_op(self, pred, labels, var_scope=None):
        with tf.device('/cpu:0'):
            # data term
            pred = tf.reshape(pred, [-1, self.num_classes])
            labels = tf.reshape(labels, [-1])
            if self.loss_func_params:
                data_loss = self.data_loss_func(pred,
                                                labels,
                                                **self.loss_func_params)
            else:
                data_loss = self.data_loss_func(pred, labels)
            if self.decay <= 0:
                return data_loss

            # regularisation term
            reg_loss = self.regularisation_loss_func(var_scope)
            return tf.add(data_loss, self.decay * reg_loss, name='total_loss')


# Generalised Dice score with different type weights
def generalised_dice_loss(pred, labels, type_weight='Square'):
    n_voxels = labels.get_shape()[0].value
    n_classes = pred.get_shape()[1].value
    pred = tf.nn.softmax(pred)
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, labels], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=[1.0] * n_voxels,
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
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, labels], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=[1.0] * n_voxels,
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
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, labels], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=[1.0] * n_voxels,
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


SUPPORTED_OPS = {"CrossEntropy": cross_entropy,
                 "Dice": dice,
                 "GDSC": generalised_dice_loss,
                 "SensSpec": sensitivity_specificity_loss}
