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
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.contrib.framework.python.framework import checkpoint_utils




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
                 omit_restore=None,
                 **_unused):
        if omit_restore is None:
            omit_restore = []
        self.initial_iter = initial_iter
        self.file_name_prefix = make_model_name(model_dir)
        self.omit_restore = omit_restore
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
        tf.logging.info('starting from iter %d', self.initial_iter)
        checkpoint = '{}-{}'.format(self.file_name_prefix, self.initial_iter)
        tf.logging.info('Accessing %s', checkpoint)
        all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)

        var_list = [v for v in all_variables]
        var_list_fin = [v for v in var_list if len([f for f in list(
            self.omit_restore) if f in v.name]) == 0]
        all_ckpt_var = checkpoint_utils.list_variables(checkpoint)
        all_names = [c[0] for c in all_ckpt_var]
        try:
            v_to_init = [v for v in all_variables if v not in var_list_fin]
            var_list_fin2 = [v for v in var_list_fin if v.name[:-2] in
                             all_names]
            saver = tf.train.Saver(save_relative_paths=True,
                                   var_list=var_list_fin2)
            saver.restore(tf.get_default_session(), checkpoint)
            var_list_fin2_names = [ v.name for v in var_list_fin2]
            v_to_init2 = [v for v in all_variables if v.name not in
                          var_list_fin2_names]
            for v in v_to_init2:
                print("Initialising ", v.name)
            init_others = tf.variables_initializer(
                v_to_init2)
            tf.get_default_session().run(init_others)
            print("Restored model")
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
                 omit_save=None,
                 **_unused):

        if omit_save is None:
            omit_save = []
        self.omit_save = omit_save
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
        all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        var_list = [v for v in all_variables if
                    "Adam" not in v.name]
        var_list = [v for v in all_variables]
        var_list_fin = [ v for v in var_list if len([f for f in list(
            self.omit_save) if f in v.name]) == 0]
        self.saver = tf.train.Saver(
            max_to_keep=self.max_checkpoints, save_relative_paths=True,
            var_list=var_list_fin)

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
