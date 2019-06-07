# -*- coding: utf-8 -*-
"""
This module implements a model checkpoint loader and writer.
"""
import os

import tensorflow as tf

from niftynet.engine.application_variables import global_vars_init_or_restore
from niftynet.engine.signal import \
    ITER_FINISHED, SESS_FINISHED, SESS_STARTED
from niftynet.io.misc_io import touch_folder

FILE_PREFIX = 'model.ckpt'


def make_model_name(model_dir):
    """
    Make the model checkpoint folder.
    the checkpoint file will be located at `model_dir/models/` folder,
    the filename will start with FILE_PREFIX.

    :param model_dir: niftynet model folder
    :return: a partial name of a checkpoint file `model_dir/model/FILE_PREFIX`
    """
    _model_dir = touch_folder(os.path.join(model_dir, 'models'))
    return os.path.join(_model_dir, FILE_PREFIX)


class ModelRestorer(object):
    """
    This class handles restoring the model at the beginning of a session.
    """

    def __init__(self,
                 model_dir,
                 initial_iter=0,
                 is_training_action=True,
                 vars_to_restore=None,
                 **_unused):
        self.initial_iter = initial_iter
        self.vars_to_restore = vars_to_restore
        self.file_name_prefix = make_model_name(model_dir)
        # randomly initialise or restoring model
        if is_training_action and initial_iter == 0:
            SESS_STARTED.connect(self.rand_init_model)
        else:
            SESS_STARTED.connect(self.restore_model)

    def rand_init_model(self, _sender, **_unused):
        """
        Randomly initialising all trainable variables defined in
        the default session.

        :param _sender:
        :param _unused:
        :return:
        """
        with tf.name_scope('Initialisation'):
            init_op = global_vars_init_or_restore()
        tf.get_default_session().run(init_op)
        tf.logging.info('Parameters from random initialisations ...')

    def restore_model(self, _sender, **_unused):
        """
        Loading checkpoint files as variable initialisations.

        :param _sender:
        :param _unused:
        :return:
        """
        checkpoint = '{}-{}'.format(self.file_name_prefix, self.initial_iter)
        to_restore = None  # tf.train.Saver's default value, restoring all

        if self.vars_to_restore:
            # partially restore (updating `to_restore` list)
            tf.logging.info("Finding variables to restore...")
            import re
            # Determine which vars to
            # restore using regex matching
            var_regex = re.compile(self.vars_to_restore)
            to_restore, to_randomise = [], []
            for restorable in tf.global_variables():
                if var_regex.search(restorable.name):
                    to_restore.append(restorable)
                else:
                    to_randomise.append(restorable)

            if not to_restore:
                tf.logging.fatal(
                    'vars_to_restore specified: %s, but nothing matched.',
                    self.vars_to_restore)
                assert to_restore, 'Nothing to restore (--vars_to_restore)'

            var_names = [  # getting first three item to print
                var_restore.name for var_restore in to_restore[:3]]
            tf.logging.info(
                'Restoring %s out of %s variables from %s: \n%s, ...',
                len(to_restore),
                len(tf.global_variables()),
                checkpoint, ',\n'.join(var_names))
            # Initialize vars to randomize
            init_op = tf.variables_initializer(to_randomise)
            tf.get_default_session().run(init_op)


        try:
            saver = tf.train.Saver(
                var_list=to_restore, save_relative_paths=True)
            saver.restore(tf.get_default_session(), checkpoint)
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


class ModelSaver(object):
    """
    This class handles iteration events to save the model as checkpoint files.
    """

    def __init__(self,
                 model_dir,
                 save_every_n=0,
                 max_checkpoints=1,
                 is_training_action=True,
                 **_unused):

        self.save_every_n = save_every_n
        self.max_checkpoints = max_checkpoints
        self.file_name_prefix = make_model_name(model_dir)
        self.saver = None

        # initialise the saver after the graph finalised
        SESS_STARTED.connect(self.init_saver)
        # save the training model at a positive frequency
        if self.save_every_n > 0:
            ITER_FINISHED.connect(self.save_model_interval)
        # always save the final training model before exiting
        if is_training_action:
            SESS_FINISHED.connect(self.save_model)

    def init_saver(self, _sender, **_unused):
        """
        Initialise a model saver.

        :param _sender:
        :param _unused:
        :return:
        """
        self.saver = tf.train.Saver(
            max_to_keep=self.max_checkpoints, save_relative_paths=True)

    def save_model(self, _sender, **msg):
        """
        Saving the model at the current iteration.

        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        iter_i = msg['iter_msg'].current_iter
        if iter_i >= 0:
            self._save_at(iter_i)

    def save_model_interval(self, _sender, **msg):
        """
        Saving the model according to the frequency of ``save_every_n``.

        :param _sender:
        :param msg: an iteration message instance
        :return:
        """
        if not msg['iter_msg'].is_training:
            return
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
