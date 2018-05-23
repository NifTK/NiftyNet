# -*- coding: utf-8 -*-
"""
This module implements a TensorBoard log writer.
"""
import os

import tensorflow as tf

from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.signal import TRAIN, VALID, ITER_STARTED, ITER_FINISHED
from niftynet.io.misc_io import get_latest_subfolder


class TensorBoardLogger(object):
    """
    This class handles iteration events to log summaries to
    the TensorBoard log.
    """

    def __init__(self,
                 model_dir=None,
                 initial_iter=0,
                 tensorboard_every_n=0,
                 **_unused):

        self.tensorboard_every_n = tensorboard_every_n
        # creating new summary subfolder if it's not finetuning
        self.summary_dir = get_latest_subfolder(
            os.path.join(model_dir, 'logs'), create_new=initial_iter == 0)
        # initialise summary writer
        if not self.summary_dir or tensorboard_every_n <= 0:
            return
        self.writer_train = tf.summary.FileWriter(
            os.path.join(self.summary_dir, TRAIN), tf.get_default_graph())
        self.writer_valid = tf.summary.FileWriter(
            os.path.join(self.summary_dir, VALID), tf.get_default_graph())

        ITER_STARTED.connect(self.read_tensorboard_op)
        ITER_FINISHED.connect(self.write_tensorboard)

    def read_tensorboard_op(self, sender, **msg):
        """
        Get TensorBoard summary_op from application at the
        beginning of each iteration.

        :param sender: a niftynet.application instance
        :param msg: should contain an IterationMessage instance
        """
        _iter_msg = msg['iter_msg']
        if _iter_msg.is_inference:
            return
        if self._is_writing(_iter_msg.current_iter):
            tf_summary_ops = sender.outputs_collector.variables(TF_SUMMARIES)
            _iter_msg.ops_to_run[TF_SUMMARIES] = tf_summary_ops

    def write_tensorboard(self, _sender, **msg):
        """
        Write to tensorboard when received the iteration finished signal.

        :param _sender:
        :param msg:
        """
        _iter_msg = msg['iter_msg']
        if not self._is_writing(_iter_msg.current_iter):
            return
        if _iter_msg.is_training:
            _iter_msg.to_tf_summary(self.writer_train)
        elif _iter_msg.is_validation:
            _iter_msg.to_tf_summary(self.writer_valid)

    def _is_writing(self, c_iter):
        """
        Decide whether to save a TensorBoard log entry for a given iteration.

        :param c_iter: Integer of the current iteration number
        :return: boolean True if is writing at the current iteration
        """
        return self.summary_dir and c_iter % self.tensorboard_every_n == 0
