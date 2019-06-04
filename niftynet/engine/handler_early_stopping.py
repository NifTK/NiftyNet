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

    def compute_generalisation_loss(validation_his):
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

        if len(_sender.performance_history) == _sender.patience:
            thresh = -1
            value = _sender.performance_history[-1]
            # Value / threshold based methods:
            # Check the latest value of the performance history against a
            # treshold calculated based on the performance history
            if _sender.mode == 'mean':
                performance_to_consider = _sender.performance_history[:-1]
                thresh = np.mean(performance_to_consider)

            if _sender.mode == 'robust_mean':
                performance_to_consider = _sender.performance_history[:-1]
                perc = np.percentile(performance_to_consider, q=[5, 95])
                temp = []
                for perf_val in performance_to_consider:
                    if perf_val < perc[1] and perf_val > perc[0]:
                        temp.append(perf_val)
                thresh = np.mean(temp)

            if value < thresh:
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
            else:
                k = 5
            s = np.floor(_sender.patience / k)


            # Median smoothing
            if _sender.mode == 'median':
                if _sender.patience%2 == 0:
                    #even patience
                    kernel_size = int(_sender.patience / 2) + 1
                else:
                    #uneven
                    kernel_size = int(np.round(_sender.patience / 2))
                thresh = signal.medfilt(performance_to_consider,
                                        kernel_size=kernel_size)

            #generalisation losses

            #actual stop check
            if value < thresh:
                msg['iter_msg'].should_stop = True
                return

        return



