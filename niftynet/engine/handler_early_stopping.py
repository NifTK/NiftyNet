# -*- coding: utf-8 -*-
"""
This module implements an early stopping handler
"""

import numpy as np
import tensorflow as tf
from scipy.ndimage import median_filter

from niftynet.engine.signal import ITER_FINISHED


class EarlyStopper(object):
    """
    This class handles iteration events to store the current performance as
    an attribute of the sender (i.e. application).
    """

    def __init__(self, **_unused):
        ITER_FINISHED.connect(self.check_criteria)

    def check_criteria(self, _sender, **msg):
        """
        Printing iteration message with ``tf.logging`` interface.
        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        msg = msg['iter_msg']
        if len(_sender.performance_history) == _sender.patience:
            # Value / threshold based methods:
            # Check the latest value of the performance history against a
            # threshold calculated based on the performance history
            msg.should_stop = \
                check_should_stop(
                    mode=_sender.mode,
                    performance_history=_sender.performance_history)


def compute_generalisation_loss(validation_his):
    """
    This function computes the generalisation loss as
        l[-1]-min(l)/max(l)-min(l)
    :param validation_his: performance history
    :return: generalisation loss
    """
    min_val_loss = np.min(np.array(validation_his))
    max_val_loss = np.max(np.array(validation_his))
    last = validation_his[-1]
    if min_val_loss == 0:
        return last
    return (last-min_val_loss)/(max_val_loss - min_val_loss)


def check_should_stop(performance_history, mode='mean', min_delta=0.03,
                      kernel_size=5, k_splits=5):
    """
    This function takes in a mode, performance_history and patience and
    returns True if the application should stop early.
    :param mode: {'mean', 'robust_mean', 'median', 'generalisation_loss',
     'median_smoothing', 'validation_up'} the default mode is 'mean'
    mean:
        If your last loss is less than the average across the entire
        performance history stop training
    robust_mean:
        Same as 'mean' but only loss values within 5th and 95th
        percentile are considered
    median:
        As in mode='mean' but using the median
    generalisation_loss:
        Computes generalisation loss over the performance
        history, and stops if it reaches an arbitrary threshold of 0.2.
    validation_up:
        This method check for performance increases in k sub-arrays of
        length s, where k x s = patience. Because we cannot guarantee
        patience to be divisible by both k and s, we define that k is
        either 4 or 5, depending on which has the smallest remainder when
        dividing.
    :param performance_history: a list of size patience with the performance
    history
    :param min_delta: threshold for smoothness
    :param kernel_size: hyperparameter for median smoothing
    :param k_splits: number of splits if using 'validation_up'
    :return:
    """
    if mode == 'mean':
        performance_to_consider = performance_history[:-1]
        thresh = np.mean(performance_to_consider)
        tf.logging.info("====Mean====")
        tf.logging.info(thresh)
        tf.logging.info(performance_history[-1])
        should_stop = performance_history[-1] > thresh

    elif mode == 'robust_mean':
        performance_to_consider = performance_history[:-1]
        perc = np.percentile(performance_to_consider, q=[5, 95])
        temp = []
        for perf_val in performance_to_consider:
            if perc[0] < perf_val < perc[1]:
                temp.append(perf_val)
        should_stop = performance_history[-1] > np.mean(temp)

    elif mode == 'median':
        performance_to_consider = performance_history[:-1]
        should_stop = performance_history[-1] > np.median(
            performance_to_consider)

    elif mode == 'generalisation_loss':
        value = compute_generalisation_loss(performance_history)
        should_stop = value > 0.2

    elif mode == 'median_smoothing':
        smoothed = median_filter(performance_history[:-1],
                                 size=kernel_size)
        gradient = np.gradient(smoothed)
        thresholded = np.where(gradient < min_delta, 1, 0)
        value = np.sum(thresholded) * 1.0 / len(gradient)
        should_stop = value < 0.5
    elif mode == 'validation_up':
        remainder = len(performance_history) % k_splits
        performance_to_consider = performance_history[remainder:]
        strips = np.split(np.array(performance_to_consider), k_splits)
        gl_increase = []
        for strip in strips:
            generalisation_loss = compute_generalisation_loss(strip)
            gl_increase.append(generalisation_loss >= min_delta)
        tf.logging.info("====Validation_up====")
        tf.logging.info(gl_increase)
        should_stop = False not in gl_increase
    else:
        raise Exception('Mode: {} provided is not supported'.format(mode))
    return should_stop
