# -*- coding: utf-8 -*-
"""
Message stores status info of the current iteration.
"""

import time

from niftynet.engine.application_variables import CONSOLE, TF_SUMMARIES
from niftynet.io.image_sets_partitioner import TRAIN, VALID, INFER
from niftynet.utilities.decorators import singleton
from niftynet.utilities.util_common import look_up_operations

CONSOLE_FORMAT = "{} iter {}, {} ({:3f}s)"
PHASES = {TRAIN, VALID, INFER}


# pylint: disable=too-many-instance-attributes
@singleton
class IterationMessage(object):
    """
    This class consists of network variables and operations at each iteration.
    A singleton instance is managed by the application engine and an
    application jointly.
    """
    _current_iter = 0
    _current_iter_tic = 0
    _current_iter_toc = 0
    _current_iter_output = None

    _data_feed_dict = None
    _ops_to_run = None
    _phase = TRAIN

    _should_stop = False

    # _callbacks = None

    @property
    def current_iter(self):
        """
        Current iteration index
        can be used to create complex schedule for the
        iterative training/validation/inference procedure.

        :return: integer of iteration
        """
        return self._current_iter

    @current_iter.setter
    def current_iter(self, value):
        self._current_iter = int(value)
        self._current_iter_tic = time.time()
        self._current_iter_output = None

    @property
    def ops_to_run(self):
        """
        operations (tf graph elements) to be fed into
        ``session.run(...)``. This is currently mainly used
        for passing network gradient updates ops to ``session.run``.

        To modify the operations, assigns ``self.ops_to_run``

        :return: a copy of the operation dictionary
        """
        if self._ops_to_run is None:
            self._ops_to_run = {}
        assert isinstance(self._ops_to_run, dict), \
            'ops to run should be a dictionary'
        return self._ops_to_run

    @ops_to_run.setter
    def ops_to_run(self, value):
        self._ops_to_run = value

    @property
    def data_feed_dict(self):
        """
        A dictionary that maps graph elements to values
        to be fed into ``session.run(...)`` as feed_dict parameter

        :return: dictionary of operations
        """
        if self._data_feed_dict is None:
            self._data_feed_dict = {}
        return self._data_feed_dict

    @data_feed_dict.setter
    def data_feed_dict(self, value):
        assert isinstance(value, dict), \
            'data_feed_dict should a dictionary of placeholders:values'
        self._data_feed_dict = value

    @property
    def current_iter_output(self):
        """
        This property stores graph output received
        by running ``session.run()``.

        :return:
        """
        return self._current_iter_output

    @current_iter_output.setter
    def current_iter_output(self, value):
        self._current_iter_output = value
        self._current_iter_toc = time.time()

    @property
    def should_stop(self):
        """
        Engine check this property after each iteration

        This could be modified in by application
        ``application.set_iteration_update()``
        to create training schedules such as early stopping.

        :return: boolean
        """
        return self._should_stop

    @should_stop.setter
    def should_stop(self, value):
        self._should_stop = bool(value)

    @property
    def phase(self):
        """
        A string indicating the phase in train/validation/inference

        :return:
        """
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = look_up_operations(value, PHASES)

    @property
    def is_training(self):
        """

        :return: boolean value indicating if the phase is in training
        """
        return self.phase == TRAIN

    @property
    def is_validation(self):
        """

        :return: boolean value indicating if the phase is validation
        """
        return self.phase == VALID

    @property
    def is_inference(self):
        """

        :return: boolean value indicating if the phase is inference
        """
        return self.phase == INFER

    @property
    def iter_duration(self):
        """
        measuring time used
        from setting self.current_iter to setting self.current_iter_output

        :return: time duration of an iteration
        """

        current_toc = max(self._current_iter_toc, self._current_iter_tic)
        return current_toc - self._current_iter_tic

    def to_console_string(self):
        """
        converting current_iter_output to string, for console displaying

        :return: summary string
        """
        summary_indentation = "    " if self.is_validation else ""
        summary_format = summary_indentation + CONSOLE_FORMAT
        console_content = ''
        try:
            console_content = self.current_iter_output.get(CONSOLE, '')
        except AttributeError:
            console_content = "print to console -- set current_iter_output " \
                              "to a dictionary of {CONSOLE: 'content'}."
        result_str = _console_vars_to_str(console_content)
        summary = summary_format.format(
            self.phase, self.current_iter, result_str, self.iter_duration)
        return summary

    def to_tf_summary(self, writer=None):
        """
        converting current_iter_output to tf summary and write to ``writer``

        :param writer: writer instance for summary output
        :return:
        """
        if writer is None:
            return
        try:
            summary = self.current_iter_output.get(TF_SUMMARIES, {})
        except AttributeError:
            summary = None

        if not summary:
            return
        writer.add_summary(summary, self.current_iter)


def _console_vars_to_str(console_dict):
    """
    Printing values of variable evaluations to command line output.
    """
    if not console_dict:
        return ''
    if isinstance(console_dict, dict):
        console_str = ', '.join('{}={}'.format(key, val)
                                for (key, val) in console_dict.items())
    else:
        console_str = '{}'.format(console_dict)
    return console_str
