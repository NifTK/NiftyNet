# -*- coding: utf-8 -*-
"""
Loss functions for multi-class classification
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.engine.application_factory import LossClassificationMultiFactory
from niftynet.layer.base_layer import Layer
#from niftynet.layer.loss_segmentation import labels_to_one_hot


class LossFunction(Layer):
    def __init__(self,
                 n_class,
                 n_rater,
                 loss_type='CrossEntropy',
                 loss_func_params=None,
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)
        self._num_classes = n_class
        self._num_raters = n_rater
        if loss_func_params is not None:
            self._loss_func_params = loss_func_params
        else:
            self._loss_func_params = {}
        self._data_loss_func = None
        self.make_callable_loss_func(loss_type)

    def make_callable_loss_func(self, type_str):
        self._data_loss_func = LossClassificationMultiFactory.create(type_str)

    def layer_op(self,
                 pred_ave=None,
                 pred_multi=None,
                 ground_truth=None,
                 weight_batch=None,
                 var_scope=None, ):
        '''
        Compute the losses in the case of a multirater setting
        :param pred_ave: average of the predictions over the different raters
        :param pred_multi: prediction for each individual rater
        :param ground_truth: ground truth classification for each individual
        rater
        :param weight_batch:
        :param var_scope:
        :return:
        '''


        with tf.device('/cpu:0'):
            if ground_truth is not None:
                ground_truth = tf.reshape(ground_truth, [-1,
                                                         self._num_raters,
                                                         self._num_classes])
            if pred_ave is not None:
                if not isinstance(pred_ave, (list, tuple)):
                    pred_ave = [pred_ave]
                if self._num_classes > 0 and pred_ave is not None:
                        # reshape the prediction to [n_voxels , num_classes]
                    pred_ave = [tf.reshape(pred, [-1, self._num_classes])
                                for pred in pred_ave]
            if pred_multi is not None and not isinstance(pred_multi, (list, \
                    tuple)):
                pred_multi = [pred_multi]
                if self._num_classes > 0 and pred_multi is not None:

                    pred_multi = [tf.reshape(pred, [-1,
                                                self._num_raters,
                                                self._num_classes])
                                    for pred in pred_multi]


            data_loss = []
            if ground_truth is not None:
                if pred_multi is not None:
                    if pred_ave is not None:
                        for pred, pred_mul in zip(pred_ave, pred_multi):
                            if self._loss_func_params:
                                data_loss.append(
                                    self._data_loss_func(ground_truth,
                                                         pred, pred_mul,
                                                         **self._loss_func_params))
                            else:
                                data_loss.append(self._data_loss_func(
                                    ground_truth, pred, pred_mul))
                    else:
                        for pred_mul in pred_multi:
                            if self._loss_func_params:
                                data_loss.append(
                                    self._data_loss_func(ground_truth,
                                                         pred_mul,
                                                         **self._loss_func_params))
                            else:
                                data_loss.append(self._data_loss_func(
                                    ground_truth, pred_mul))
                else:
                    for pred in pred_ave:

                        if self._loss_func_params:
                            data_loss.append(self._data_loss_func(
                                pred, ground_truth,
                                **self._loss_func_params))
                        else:
                            data_loss.append(self._data_loss_func(
                                pred, ground_truth))
            elif pred_multi is not None:
                for pred, pred_mul in zip(pred_ave, pred_multi):
                    if self._loss_func_params:
                        data_loss.append(self._data_loss_func(
                            pred, pred_mul,
                            **self._loss_func_params))
                    else:
                        data_loss.append(self._data_loss_func(
                            pred, pred_mul))

            if weight_batch is not None:
                return tf.reduce_mean(weight_batch/tf.reduce_sum(
                    weight_batch) * data_loss[0])
            else:
                return tf.reduce_mean(data_loss)

#
def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.

    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end) and the output shape
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
        print('no need')
        return tf.reshape(ground_truth, output_shape), output_shape

    # squeeze the spatial shape

    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0],
                            num_classes_tf], 0)
    dense_shape = tf.Print(tf.cast(dense_shape, tf.int64), [dense_shape,
                           output_shape], message='check_shape_lohe')
    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(tf.shape(ground_truth)[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot, output_shape


def loss_confusion_matrix(ground_truth, pred_multi, num_classes=2, nrater=6):
    '''
    Creates a loss over the two multi rater confusion matrices between the rater
    :param ground_truth: multi rater classification
    :param pred_multi: multi rater prediction (1 pred per class for each
    rater and each observation - A softmax is performed during the loss
    calculation
    :param nrater: number of raters
    :return: integration over the absolute differences between the confusion
    matrices divided by number of raters
    '''

    one_hot_gt, output_shape = labels_to_one_hot(ground_truth, num_classes)
    dense_one_hot = tf.reshape(tf.sparse_tensor_to_dense(one_hot_gt),
                               output_shape)
    dense_one_hot = tf.reshape(dense_one_hot, tf.shape(pred_multi))
    nclasses=tf.shape(pred_multi)[-1]
    nn_pred = tf.nn.softmax(pred_multi,-1)
    error_fin = tf.zeros([nclasses, nclasses])
    error_fin = tf.Print(tf.cast(error_fin, tf.float32), [nn_pred, tf.shape(
        pred_multi)], message='error')
    nn_pred = tf.Print(tf.cast(nn_pred, tf.float32), [tf.shape(
        dense_one_hot), nclasses, tf.shape(ground_truth), tf.shape(nn_pred)],
                       message='check_conf')
    for i in range(0, nrater):
        for j in range(i+1, nrater):

            confusion_pred = tf.matmul(tf.transpose(nn_pred[:, i, :]),
                                       nn_pred[:, j, :])
            confusion_gt = tf.matmul(tf.transpose(dense_one_hot[:, i, :]),
                                     dense_one_hot[:, j, :])
            error = tf.divide(tf.abs(confusion_gt - confusion_pred), tf.cast(
                tf.shape(ground_truth)[0], tf.float32))
            error_fin += error
            error_fin = tf.Print(tf.cast(error,tf.float32), [tf.reduce_sum(
                error_fin), tf.reduce_max(error_fin)], message='build_error')
    return tf.reduce_sum(error_fin)/tf.cast(nrater, tf.float32)


def variability(pred_multi, num_classes=2, nrater=2):
    one_hot_gt, output_shape = labels_to_one_hot(tf.cast(pred_multi, tf.int64),
                                           num_classes)
    dense_one_hot = tf.sparse_tensor_to_dense(one_hot_gt)
    freq = tf.divide(tf.reduce_sum(dense_one_hot, 1), tf.cast(tf.shape(
        pred_multi)[1],tf.float32))
    variability = tf.reduce_sum(tf.square(freq), -1)
    return 1 - variability


def loss_variability(ground_truth, pred_multi, weight_map=None):
    one_hot_gt, output_shape = labels_to_one_hot(tf.cast(ground_truth,
                                                         tf.int64),
                                   tf.shape(pred_multi)[-1])
    dense_gt = tf.sparse_tensor_to_dense(one_hot_gt)
    pred_hard = tf.argmax(pred_multi, -1)

    one_hot_pred, _ = labels_to_one_hot(tf.cast(pred_hard, tf.int64),
                                   tf.shape(pred_multi)[-1])
    dense_pred = tf.sparse_tensor_to_dense(one_hot_pred)
    freq_pred = tf.divide(tf.reduce_sum(dense_pred, 1),
                          tf.cast(tf.shape(pred_multi)[1],tf.float32))
    variability_pred = tf.reduce_sum(tf.square(freq_pred), -1)
    freq_gt = tf.divide(tf.reduce_sum(dense_gt, 1),
                        tf.cast(tf.shape(pred_multi)[1],tf.float32))
    variability_gt = tf.reduce_sum(tf.square(freq_gt), -1)

    diff_square = tf.square(variability_gt-variability_pred)
    if weight_map is not None:
        diff_square = weight_map * diff_square
    loss = tf.sqrt(tf.reduce_mean(diff_square))
    return loss


def rmse_consistency(pred_ave,
                     pred_multi, weight_map=None):
    pred_multi  = tf.nn.softmax(pred_multi, -1)
    pred_multi_ave = tf.reduce_mean(pred_multi, axis=1)
    pred_multi_ave = tf.Print(tf.cast(pred_multi_ave, tf.float32), [pred_ave[0],
                                                        pred_multi_ave[0],
                                                                    tf.shape(
                                                                        pred_ave), tf.shape(pred_multi_ave),
                              tf.reduce_max(pred_ave-pred_multi_ave)],
                              message='rmse_test')
    diff_square = tf.square(pred_ave-pred_multi_ave)
    if weight_map is not None:
        diff_square = tf.multiply(weight_map, diff_square) / tf.reduce_sum(
            weight_map)

    return tf.sqrt(tf.reduce_mean(diff_square))


