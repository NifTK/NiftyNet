# -*- coding: utf-8 -*-
"""
This module implements a model checkpoint writer.
"""

import tensorflow as tf

from niftynet.engine.signal import ITER_FINISHED, SESS_FINISHED


class ModelSaver(object):
    """
    This class handles iteration events to save the model as checkpoint files.
    """

    def __init__(self,
                 save_every_n,
                 session_prefix,
                 saver,
                 is_training,
                 **_unused):

        self.save_every_n = save_every_n
        self.file_name_prefix = session_prefix
        self.saver = saver

        if is_training:
            SESS_FINISHED.connect(self.save_model)
            if save_every_n > 0:
                ITER_FINISHED.connect(self.save_model_interval)

    def save_model(self, _sender, **msg):
        """
        Saving the model at the current iteration.
        :param _sender:
        :param msg: an iteration message instance
        """
        _iter_msg = msg.get('iter_msg', None)
        if _iter_msg is not None:
            self._save_at(_iter_msg.current_iter)

    def save_model_interval(self, _sender, **msg):
        """
        Saving the model according to the frequency of ``save_every_n``.
        :param _sender:
        :param msg: an iteration message instance
        """
        _iter_msg = msg.get('iter_msg', None)
        if _iter_msg is None:
            return
        iter_i = _iter_msg.current_iter
        if iter_i > 0 and iter_i % self.save_every_n == 0:
            self._save_at(iter_i)

    def _save_at(self, iter_i):
        """
        Saving the model at iter i and print a console log.
        :param iter_i: integer of the current iteration
        """
        self.saver.save(sess=tf.get_default_session(),
                        save_path=self.file_name_prefix,
                        global_step=iter_i)
        tf.logging.info(
            'iter %d saved: %s', iter_i, self.file_name_prefix)
