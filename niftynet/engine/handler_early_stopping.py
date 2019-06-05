from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.signal import ITER_FINISHED

import tensorflow as tf
import numpy as np
from scipy import signal



class EarlyStopper(object):
    """
    This class handles iteration events to store the current performance as
    an attribute of the sender (i.e. application).
    """

    def __init__(self, **_unused):
        ITER_FINISHED.connect(self.check_criteria)

    def compute_generalisation_loss(self, validation_his):
        min_val_loss = np.min(np.array(validation_his))
        last = validation_his[-1]
        return np.divide(last, min_val_loss) - 1.0

    def check_criteria(self, _sender, **msg):
        """
        Printing iteration message with ``tf.logging`` interface.
        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        msg = msg['iter_msg']
        min_delta = -1
        #Min delta is a amount of change we ignore.

        if len(_sender.performance_history) == _sender.patience:
            tresh = None
            value = _sender.performance_history[-1]

            # Value / treshold based methods:
            # Check the latest value of the performance history against a
            # treshold calculated based on the performance history
            if _sender.mode == 'mean':
                performance_to_consider = _sender.performance_history[:-1]
                tresh = np.mean(performance_to_consider)

            elif _sender.mode == 'robust_mean':
                performance_to_consider = _sender.performance_history[:-1]
                perc = np.percentile(performance_to_consider, q=[5, 95])
                temp = []
                for perf_val in performance_to_consider:
                    if perf_val < perc[1] and perf_val > perc[0]:
                        temp.append(perf_val)
                tresh = np.mean(temp)

            elif _sender.mode == 'median':
                performance_to_consider = _sender.performance_history[:-1]
                tresh = np.median(performance_to_consider)

            elif _sender.mode == 'std':
                value = np.std(_sender.performance_history)
                tresh = 0.01

            elif _sender.mode == 'robust_std':
                tresh = 0.01
                perc = np.percentile(_sender.performance_history, q=[5, 95])
                temp = []
                for perf_val in _sender.performance_history:
                    if perf_val < perc[1] and perf_val > perc[0]:
                        temp.append(perf_val)
                value = np.std(temp)

            elif _sender.mode == 'generalisation_loss':
                value = self.compute_generalisation_loss(
                    _sender.performance_history[:-1])
                tresh = 0.2

            elif _sender.mode == 'median_smoothing':
                if _sender.patience%2 == 0:
                    #even patience
                    kernel_size = int(_sender.patience / 2) + 1
                else:
                    #uneven
                    kernel_size = int(np.round(_sender.patience / 2))
                smoothed = signal.medfilt(performance_to_consider,
                                          kernel_size=kernel_size)
                gradient = np.gradient(smoothed)
                tresholded = np.where(np.abs(gradient) < 0.03, 1, 0)
                value = np.sum(tresholded) / len(gradient)

            #actual stop check
            if tresh is not None and value < tresh:
                msg['iter_msg'].should_stop = True
                return

            # Strip-based methods:
            # These methods check for performance increases in k sub-arrays of
            # length s, where k x s = patience. Because we cannot guarantee
            # patience to be divisible by both k and s, we define that k is
            # either 4 or 5, depending on which has the smallest remainder when
            # dividing.
            remainder_5 = _sender.patience % 5
            remainder_4 = _sender.patience % 4

            if remainder_4 < remainder_5:
                k = 4
                remainder = remainder_4
            else:
                k = 5
                remainder = remainder_5
            s = np.floor(_sender.patience / k)
            performance_to_consider = _sender.performance_history[remainder:]

            strips = np.split(np.array(performance_to_consider), k)


            if _sender.mode == 'validation_up':
                GL_increase = []
                for strip in strips:
                    GL = self.compute_generalisation_loss(
                        strip)
                    GL_increase.append(GL > (0 + min_delta))
                if GL_increase.__contains__(False):
                    return
                else:
                    msg['iter_msg'].should_stop = True

        return



