# -*- coding: utf-8 -*-
"""
This module implements a model checkpoint writer.
"""
import os
import tensorflow as tf

from niftynet.engine.signal import ITER_FINISHED, SESS_STARTED, SESS_FINISHED
from niftynet.engine.application_variables import global_vars_init_or_restore


FILE_PREFIX = 'model.ckpt'


class ModelSaver(object):
    """
    This class handles iteration events to save the model as checkpoint files.
    """

    def __init__(self,
                 model_dir,
                 initial_iter=0,
                 max_checkpoints=1,
                 save_every_n=0,
                 is_training=True,
                 **_unused):

        self.model_dir = model_dir
        self.initial_iter = initial_iter
        self.save_every_n = save_every_n
        self.is_training = is_training

        self.file_name_prefix = os.path.join(self.model_dir, FILE_PREFIX)
        self.saver = tf.train.Saver(
            max_to_keep=max_checkpoints, save_relative_paths=True)

        SESS_STARTED.connect(self.rand_init_or_restore_vars)
        if self.is_training:
            ITER_FINISHED.connect(self.save_model_interval)
            SESS_FINISHED.connect(self.save_model)

    def rand_init_or_restore_vars(self, _sender, **_unused):
        """
        Randomly initialising all trainable variables defined in session,
        or loading checkpoint files as variable initialisations.
        """
        tf.logging.info('starting from iter %d', self.initial_iter)
        if self.is_training and self.initial_iter == 0:
            tf.get_default_session().run(global_vars_init_or_restore())
            tf.logging.info('Parameters from random initialisations ...')
            return

        # check model's folder
        assert os.path.exists(self.model_dir), \
            "Model folder not found {}, please check" \
            "config parameter: model_dir".format(self.model_dir)
        # check model's file
        ckpt_state = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt_state is None:
            tf.logging.warning(
                "%s/checkpoint not found, please check "
                "config parameter: model_dir", self.model_dir)
        if self.initial_iter > 0:
            checkpoint = '{}-{}'.format(
                self.file_name_prefix, self.initial_iter)
        else:  # initial iter smaller than zero
            try:
                checkpoint = ckpt_state.model_checkpoint_path
                assert checkpoint, 'checkpoint path not found ' \
                                   'in {}/checkpoints'.format(self.model_dir)
                self.initial_iter = int(checkpoint.rsplit('-')[-1])
                tf.logging.info('set initial_iter to %d based '
                                'on checkpoints', self.initial_iter)
            except (ValueError, AttributeError):
                tf.logging.fatal(
                    'failed to get iteration number '
                    'from checkpoint path, please set '
                    'inference_iter or starting_iter to a positive integer')
                raise
        # restore session
        tf.logging.info('Accessing %s ...', checkpoint)
        try:
            self.saver.restore(tf.get_default_session(), checkpoint)
        except tf.errors.NotFoundError:
            tf.logging.fatal(
                'checkpoint %s not found or variables to restore do not '
                'match the current application graph', checkpoint)
            raise

    def save_model(self, _sender, **msg):
        """
        Saving the model at the current iteration.

        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        _iter_msg = msg.get('iter_msg', None)
        if _iter_msg is not None:
            self._save_at(_iter_msg.current_iter)

    def save_model_interval(self, _sender, **msg):
        """
        Saving the model according to the frequency of ``save_every_n``.

        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        _iter_msg = msg.get('iter_msg', None)
        if _iter_msg is None:
            return
        iter_i = _iter_msg.current_iter
        if iter_i > 0 and self.save_every_n > 0 \
                and iter_i % self.save_every_n == 0:
            self._save_at(iter_i)

    def _save_at(self, iter_i):
        """
        Saving the model at iter i and print a console log.

        :param iter_i: integer of the current iteration
        :return:
        """
        self.saver.save(sess=tf.get_default_session(),
                        save_path=self.file_name_prefix,
                        global_step=iter_i)
        tf.logging.info('iter %d saved: %s', iter_i, self.file_name_prefix)
