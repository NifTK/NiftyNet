# -*- coding: utf-8 -*-
"""
This module implements a TensorBoard log writer.
"""
import os

import tensorflow as tf

from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.signal import TRAIN, VALID, ITER_STARTED, ITER_FINISHED


class TensorBoardLogger(object):
    """
    This class handles iteration events to log summaries to
    the TensorBoard log.
    """

    def __init__(self,
                 summary_dir,
                 outputs_collector,
                 tensorboard_every_n=0,
                 **_unused):
        self.tensorboard_every_n = tensorboard_every_n
        self.summary_dir = summary_dir
        # the collector provides TF summary ops
        self.outputs_collector = outputs_collector
        # initialise summary writer
        if not tf.get_default_graph() or not self.summary_dir:
            return
        self.writer_train = tf.summary.FileWriter(
            os.path.join(self.summary_dir, TRAIN), tf.get_default_graph())
        self.writer_valid = tf.summary.FileWriter(
            os.path.join(self.summary_dir, VALID), tf.get_default_graph())

        ITER_STARTED.connect(self.read_tensorboard_op)
        ITER_FINISHED.connect(self.write_tensorboard)

    def read_tensorboard_op(self, _sender, **msg):
        """
        Get TensorBoard summary_op from application at the
        beginning of each iteration.

        :param _sender: signal
        :param msg: should contain an IterationMessage instance
        """
        _iter_msg = msg.get('iter_msg', None)
        if _iter_msg is None:
            return
        if _iter_msg.is_inference:
            return
        if self._is_writing(_iter_msg.current_iter):
            tf_summary_ops = self.outputs_collector.variables(TF_SUMMARIES)
            _iter_msg.ops_to_run[TF_SUMMARIES] = tf_summary_ops

    def write_tensorboard(self, _sender, **msg):
        """
        Write to tensorboard when received the iteration finished signal.

        :param _sender:
        :param msg:
        """
        _iter_msg = msg.get('iter_msg', None)
        if _iter_msg is None or not self._is_writing(_iter_msg.current_iter):
            return
        if _iter_msg.is_training:
            _iter_msg.to_tf_summary(self.writer_train)
        elif _iter_msg.is_validation:
            _iter_msg.to_tf_summary(self.writer_valid)
        return

    def _is_writing(self, current_iter):
        """
        Decide whether to save a TensorBoard log entry for a given iteration.

        :param current_iter: Integer of the current iteration number
        :return: boolean True if is writing at the current iteration
        """
        return self.tensorboard_every_n > 0 and \
            current_iter % self.tensorboard_every_n == 0
