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

    def __init__(self, **_unused):
        ITER_STARTED.connect(self.set_tensors_to_run)
        ITER_FINISHED.connect(self.interpret_output)

    @staticmethod
    def set_tensors_to_run(sender, **msg):
        """
        Event handler to add all tensors to evaluate to the iteration message.

        :param sender: a niftynet.application instance
        :param msg: an iteration message instance
        :return:
        """
        _iter_msg = msg['iter_msg']
        _iter_msg.ops_to_run[NETWORK_OUTPUT] = \
            sender.outputs_collector.variables(NETWORK_OUTPUT)

        if _iter_msg.is_training:
            _iter_msg.data_feed_dict[sender.is_validation] = False
        elif _iter_msg.is_validation:
            _iter_msg.data_feed_dict[sender.is_validation] = True

        sender.set_iteration_update(_iter_msg)

    @staticmethod
    def interpret_output(sender, **msg):
        """
        Calling sender application to interpret evaluated tensors.
        Set _iter_msg.should_stop to a True value
        if it's an end of the engine loop.

        See also:
        ``niftynet.engine.application_driver.loop``

        :param sender: a niftynet.application instance
        :param msg: an iteration message instance
        :return:
        """
        _iter_msg = msg['iter_msg']
        waiting_for_more_output = sender.interpret_output(
            _iter_msg.current_iter_output[NETWORK_OUTPUT])
        if not waiting_for_more_output:
            _iter_msg.should_stop = OutputInterpreter.__name__
