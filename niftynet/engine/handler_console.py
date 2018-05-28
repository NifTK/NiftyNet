# -*- coding: utf-8 -*-
"""
This module implements a console output writer.
"""

import tensorflow as tf

from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.signal import ITER_STARTED, ITER_FINISHED


class ConsoleLogger(object):
    """
    This class handles iteration events to print output to the console.
    """

    def __init__(self, **_unused):
        ITER_STARTED.connect(self.read_console_vars)
        ITER_FINISHED.connect(self.print_console_vars)

    def read_console_vars(self, sender, **msg):
        """
        Event handler to add all console output ops to the iteration message

        :param sender: a niftynet.application instance
        :param msg: an iteration message instance
        :return:
        """
        msg['iter_msg'].ops_to_run[CONSOLE] = \
            sender.outputs_collector.variables(CONSOLE)

    def print_console_vars(self, _sender, **msg):
        """
        Printing iteration message with ``tf.logging`` interface.

        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        tf.logging.info(msg['iter_msg'].to_console_string())
