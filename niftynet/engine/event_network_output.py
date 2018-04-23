# -*- coding: utf-8 -*-
"""
This module implements a network output interpreter.
"""

from niftynet.engine.application_variables import NETWORK_OUTPUT
from niftynet.engine.signal import ITER_STARTED, ITER_FINISHED


class OutputInterpreter(object):
    """
    This class handles iteration events to interpret output.
    """

    def __init__(self, app, outputs_collector, **_unused):
        self.outputs_collector = outputs_collector
        self.app = app

        ITER_STARTED.connect(self.set_tensors_to_run)
        ITER_FINISHED.connect(self.interpret_output)

    def set_tensors_to_run(self, _sender, **msg):
        """
        Event handler to add all tensors to evaluate to the iteration message.

        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        _iter_msg = msg.get('iter_msg', None)
        if _iter_msg is None:
            return

        _iter_msg.ops_to_run[NETWORK_OUTPUT] = \
            self.outputs_collector.variables(NETWORK_OUTPUT)

        self.app.set_iteration_update(_iter_msg)
        # if _iter_msg.is_training:
        #    _iter_msg.data_feed_dict[self.is_validation] = False
        # elif _iter_msg.is_validation:
        #    _iter_msg.data_feed_dict[self.is_validation] = True

    def interpret_output(self, _sender, **msg):
        """
        Calling self.app to interpret evaluated tensors.
        Set _iter_msg.should_stop to a True value
        if it's an end of the engine loop.

        See also:
        ``niftynet.engine.application_driver._loop``

        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        _iter_msg = msg.get('iter_msg', None)
        if _iter_msg is None:
            return

        waiting_for_more_output = self.app.interpret_output(
            _iter_msg.current_iter_output[NETWORK_OUTPUT])
        if not waiting_for_more_output:
            _iter_msg.should_stop = type(self).__name__
