# -*- coding: utf-8 -*-
"""
Loss functions for multi-class classification
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.engine.application_factory import LossClassificationFactory
from niftynet.layer.base_layer import Layer


class LossFunction(Layer):
    def __init__(self,
                 n_class,
                 loss_type='CrossEntropy',
                 loss_func_params=None,
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)
        self._num_classes = n_class
        if loss_func_params is not None:
            self._loss_func_params = loss_func_params
        else:
            self._loss_func_params = {}
        self._data_loss_func = None
        self.make_callable_loss_func(loss_type)

    def make_callable_loss_func(self, type_str):
        self._data_loss_func = LossClassificationFactory.create(type_str)

    def layer_op(self,
                 prediction,
                 ground_truth=None,
                 var_scope=None, ):
        """
        Compute loss from `prediction` and `ground truth`,

        if `prediction `is list of tensors, each element of the list
        will be compared against `ground_truth`.

        :param prediction: input will be reshaped into (N, num_classes)
        :param ground_truth: input will be reshaped into (N,)
        :param var_scope:
        :return:
        """

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

            data_loss = []
            for pred in prediction:
                if self._loss_func_params:
                    data_loss.append(self._data_loss_func(
                        pred, ground_truth,
                        **self._loss_func_params))
                else:
                    data_loss.append(self._data_loss_func(
                        pred, ground_truth))
            return tf.reduce_mean(data_loss)


def cross_entropy(prediction,
                  ground_truth):
    """
    Function to calculate the cross entropy loss
    :param prediction: the logits (before softmax)
    :param ground_truth: the classification ground truth
    :return: the loss
    """
    ground_truth = tf.to_int64(ground_truth)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=ground_truth)
    return loss
