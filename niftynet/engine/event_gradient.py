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

    def __init__(self, app, **_unused):
        self.app = app
        ITER_STARTED.connect(self.add_gradients)

    def add_gradients(self, _sender, **msg):
        """
        Event handler to add gradients to itermsg ops_to_run.

        See also
        ``niftynet.application.base_application.set_network_gradient_op``

        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        _iter_msg = msg.get('iter_msg', None)
        if _iter_msg is None:
            return
        if _iter_msg.is_training:
            _iter_msg.ops_to_run['gradients'] = self.app.gradient_op
