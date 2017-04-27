# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class LossFunction(object):
    def __init__(self, n_class, loss_type='Dice', reg_type='L2', decay=0.0):
        self.num_classes = n_class
        self.set_loss_type(loss_type)
        self.set_regularisation_type(reg_type)
        self.set_decay(decay)
        print('Training loss: {}_loss + ({})*{}_loss'.format(
            loss_type, decay, reg_type))

    def set_loss_type(self, type_str):
        accepted_functions = {"CrossEntropy": cross_entropy,
                              "Dice": dice,
                              "GDSC": GDSC_loss,
                              "SensSpec": sensitivity_specificity_loss}
        if type_str in accepted_functions.keys():
            self.data_loss_fun = accepted_functions[type_str]
        else:
            edit_distances = {}
            for loss_name in accepted_functions.keys():
                edit_distance = damerau_levenshtein_distance(loss_name, type_str)
                if edit_distance <= 3:
                    edit_distances[loss_name] = edit_distance
            if edit_distances:
                guess_at_correct_spelling = min(edit_distances, key=edit_distances.get)
                raise ValueError(('By "{0}", did you mean "{1}"?\n '
                                  '"{0}" is not a valid loss.').format(type_str, guess_at_correct_spelling))
            else:
                raise ValueError('Loss type "%s" is not found.' % type_str)

    def set_regularisation_type(self, type_str):
        if type_str == "L2":
            self.reg_loss_fun = l2_reg_loss

    def set_decay(self, decay):
        self.decay = decay

    def total_loss(self, pred, labels, var_scope):
        with tf.device('/cpu:0'):
            # data term
            pred = tf.reshape(pred, [-1, self.num_classes])
            labels = tf.reshape(labels, [-1])
            data_loss = self.data_loss_fun(pred, labels)
            if self.decay <= 0:
                return data_loss

            # regularisation term
            reg_loss = self.reg_loss_fun(var_scope)
            return tf.add(data_loss, self.decay * reg_loss, name='total_loss')


# Generalised Dice score with different type weights
def GDSC_loss(pred, labels, type_weight='Square'):
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
        raise ValueError('The variable type_weight "%s" is not defined.' % type_weight)
    GDSC = tf.reduce_sum(tf.multiply(weights, intersect)) / tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol))
    return 1 - GDSC


# Sensitivity Specificity loss function adapted to work for multiple labels
def sensitivity_specificity_loss(pred, labels, r=0.05):
    n_voxels = labels.get_shape()[0].value
    n_classes = pred.get_shape()[1].value
    pred = tf.nn.softmax(pred)
    ids = tf.constant(np.arange(n_voxels), dtype=tf.int64)
    ids = tf.stack([ids, labels], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=[1.0] * n_voxels,
                              dense_shape=[n_voxels, n_classes])
    one_hotB = 1 - tf.sparse_tensor_to_dense(one_hot)
    SensSpec = tf.reduce_mean(
        tf.add(tf.multiply(r, tf.reduce_sum(tf.multiply(tf.square(-1 * tf.sparse_add(-1 * pred, one_hot)) \
                                                        , tf.sparse_tensor_to_dense(one_hot)),
                                            0) / tf.sparse_reduce_sum(one_hot, 0)), \
               tf.multiply((1 - r), tf.reduce_sum(tf.multiply(tf.square(-1 * tf.sparse_add(-1 * pred, one_hot)), \
                                                              one_hotB), 0) / tf.reduce_sum(one_hotB, 0))))
    return SensSpec


def l2_reg_loss(scope):
    if not tf.get_collection('reg_var', scope):
        return 0.0
    return tf.add_n([tf.nn.l2_loss(reg_var) for reg_var in
                     tf.get_collection('reg_var', scope)])


def cross_entropy(pred, labels):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels)
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
    dice_numerator = 2 * tf.sparse_reduce_sum(one_hot * pred, reduction_axes=[0])
    dice_denominator = (tf.reduce_sum(tf.square(pred), reduction_indices=[0]) +
                        tf.sparse_reduce_sum(one_hot, reduction_axes=[0]))
    epsilon_denominator = 0.00001

    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    dice_score.set_shape([n_classes])
    # minimising (1 - dice_coefficients)
    return 1.0 - tf.reduce_mean(dice_score)


def damerau_levenshtein_distance(s1, s2):
    """Calculates an edit distance, for typo detection. Code based on : 
    https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance"""

    d = {}
    string_1_length = len(s1)
    string_2_length = len(s2)
    for i in range(-1, string_1_length + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, string_2_length + 1):
        d[(-1, j)] = j + 1

    for i in range(string_1_length):
        for j in range(string_2_length):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[string_1_length - 1, string_2_length - 1]
