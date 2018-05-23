# -*- coding: utf-8 -*-
"""
This module implements a model checkpoint writer.
"""
import os

import tensorflow as tf

from niftynet.engine.application_variables import global_vars_init_or_restore
from niftynet.engine.signal import \
    ITER_FINISHED, SESS_FINISHED, GRAPH_FINALISING
from niftynet.io.misc_io import infer_latest_model_file, touch_folder

FILE_PREFIX = 'model.ckpt'


class ModelSaver(object):
    """
    This class handles iteration events to save the model as checkpoint files.
    """

    def __init__(self,
                 model_dir,
                 initial_iter=0,
                 save_every_n=0,
                 max_checkpoints=1,
                 is_training_action=True,
                 **_unused):

        self.initial_iter = initial_iter
        self.save_every_n = save_every_n
        self.max_checkpoints = max_checkpoints
        self.is_training_action = is_training_action

        self.model_dir = touch_folder(os.path.join(model_dir, 'models'))
        self.file_name_prefix = os.path.join(self.model_dir, FILE_PREFIX)
        self.saver = None

        # randomly initialise or restoring model
        if self.is_training_action and self.initial_iter == 0:
            GRAPH_FINALISING.connect(self.rand_init_model)
        else:
            # infer the initial iteration from model files
            if self.initial_iter < 0:
                self.initial_iter = infer_latest_model_file(self.model_dir)
            GRAPH_FINALISING.connect(self.restore_model)

        # save the training model at a positive frequency
        if self.save_every_n > 0:
            ITER_FINISHED.connect(self.save_model_interval)

        # always save the final training model before exiting
        if self.is_training_action:
            SESS_FINISHED.connect(self.save_model)

    def rand_init_model(self, _sender, **_unused):
        """
        Randomly initialising all trainable variables defined in
        the default session.

        :param _sender:
        :param _unused:
        :return:
        """
        self.saver = tf.train.Saver(
            max_to_keep=self.max_checkpoints, save_relative_paths=True)
        with tf.name_scope('Initialisation'):
            init_op = global_vars_init_or_restore()
        tf.Graph.finalize(tf.get_default_graph())
        tf.get_default_session().run(init_op)
        tf.logging.info('Parameters from random initialisations ...')

    def restore_model(self, _sender, **_unused):
        """
        Loading checkpoint files as variable initialisations.

        :param _sender:
        :param _unused:
        :return:
        """
        self.saver = tf.train.Saver(
            max_to_keep=self.max_checkpoints, save_relative_paths=True)
        tf.Graph.finalize(tf.get_default_graph())
        tf.logging.info('starting from iter %d', self.initial_iter)
        checkpoint = '{}-{}'.format(self.file_name_prefix, self.initial_iter)
        tf.logging.info('Accessing %s ...', checkpoint)
        try:
            self.saver.restore(tf.get_default_session(), checkpoint)
        except tf.errors.NotFoundError:
            tf.logging.fatal(
                'checkpoint %s not found or variables to restore do not '
                'match the current application graph', checkpoint)
            dir_name = os.path.dirname(checkpoint)
            if dir_name and not os.path.exists(dir_name):
                tf.logging.fatal(
                    "Model folder not found %s, please check"
                    "config parameter: model_dir", dir_name)
            raise

    def save_model(self, _sender, **msg):
        """
        Saving the model at the current iteration.

        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        if not msg.get('iter_msg', None):
            return
        if msg['iter_msg'].current_iter >= 0:
            self._save_at(msg['iter_msg'].current_iter)

    def save_model_interval(self, _sender, **msg):
        """
        Saving the model according to the frequency of ``save_every_n``.

        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        iter_i = msg['iter_msg'].current_iter
        if iter_i > 0 and iter_i % self.save_every_n == 0:
            self._save_at(iter_i)

    def _save_at(self, iter_i):
        """
        Saving the model at iter i and print a console log.

        : param iter_i: integer of the current iteration
        : return:
        """
        if not self.saver:
            return
        self.saver.save(sess=tf.get_default_session(),
                        save_path=self.file_name_prefix,
                        global_step=iter_i)
        tf.logging.info('iter %d saved: %s', iter_i, self.file_name_prefix)
