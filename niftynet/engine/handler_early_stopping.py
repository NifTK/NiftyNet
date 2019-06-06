from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.signal import ITER_FINISHED

import tensorflow as tf
import numpy as np
from scipy.ndimage import median_filter


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
                    performance_history=_sender.performance_history,
                    patience=_sender.patience)


def compute_generalisation_loss(validation_his):
    min_val_loss = np.min(np.array(validation_his))
    max_val_loss = np.max(np.array(validation_his))
    last = validation_his[-1]
    if min_val_loss == 0:
        return last
    return (last-min_val_loss)/(max_val_loss - min_val_loss)


def check_should_stop(performance_history, patience,
                      mode='mean', min_delta=0.03, kernel_size=5):
    """
    This function takes in a mode, performance_history and patience and
    returns True if the application should stop early.
    :param mode: {'mean', 'robust_mean', 'median', 'generalisation_loss', 'median_smoothing', 'validation_up'}
           the default mode is 'mean'
    :param performance_history: a list of size patience with the performance history
    :param patience: see above
    :param min_delta: threshold for smoothness
    :param kernel_size: hyperparameter for median smoothing
    :return:
    """
    if mode == 'mean':
        """
        If your last loss is less than the average across the entire 
        performance history stop training
        """
        performance_to_consider = performance_history[:-1]
        tresh = np.mean(performance_to_consider)
        should_stop = performance_history[-1] > tresh

    elif mode == 'robust_mean':
        """
        Same as 'mean' but only loss values within 5th and 95th percentile
        are considered
        """
        performance_to_consider = performance_history[:-1]
        perc = np.percentile(performance_to_consider, q=[5, 95])
        temp = []
        for perf_val in performance_to_consider:
            if perc[0] < perf_val < perc[1]:
                temp.append(perf_val)
        should_stop = performance_history[-1] > np.mean(temp)

    elif mode == 'median':
        """
        As in mode='mean' but using the median
        """
        performance_to_consider = performance_history[:-1]
        should_stop = performance_history[-1] > np.median(
            performance_to_consider)

    elif mode == 'generalisation_loss':
        """
        Computes generalisation loss over the performance history,
        and stops if it reaches an arbitrary threshold of 0.2.
        """

        value = compute_generalisation_loss(performance_history)
        should_stop = value > 0.2

    elif mode == 'median_smoothing':
        smoothed = median_filter(performance_history[:-1],
                                  size=kernel_size)
        gradient = np.gradient(smoothed)
        tresholded = np.where(gradient < min_delta, 1, 0)
        value = np.sum(tresholded) / len(gradient)
        should_stop = value < 0.5
    elif mode == 'validation_up':
        # Strip-based methods:
        # These methods check for performance increases in k sub-arrays of
        # length s, where k x s = patience. Because we cannot guarantee
        # patience to be divisible by both k and s, we define that k is
        # either 4 or 5, depending on which has the smallest remainder when
        # dividing.
        remainder = len(performance_history) % kernel_size
        performance_to_consider = performance_history[remainder:]

        strips = np.split(np.array(performance_to_consider), kernel_size)
        GL_increase = []
        for strip in strips:
            GL = compute_generalisation_loss(strip)
            GL_increase.append(GL >= min_delta)
        should_stop = False not in GL_increase
    else:
        raise Exception('Mode: {} provided is not supported'.format(mode))
    return should_stop