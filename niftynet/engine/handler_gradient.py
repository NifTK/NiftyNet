# -*- coding: utf-8 -*-
"""
This module implements a network model updater with gradient ops.
"""

from niftynet.engine.signal import ITER_STARTED


class ApplyGradients(object):
    """
    This class handles iteration events to update the model with gradient op
    (by setting iteration message with a 'gradients' op at the beginning of
    each iteration).
    """

    def __init__(self, **_unused):
        ITER_STARTED.connect(self.add_gradients)

    @staticmethod
    def add_gradients(sender, **msg):
        """
        Event handler to add gradients to iteration message ops_to_run.

        See also
        ``niftynet.application.base_application.set_network_gradient_op``

        :param sender: a niftynet.application instance
        :param msg: an iteration message instance
        :return:
        """
        if msg['iter_msg'].is_training:
            msg['iter_msg'].ops_to_run['gradients'] = sender.gradient_op
