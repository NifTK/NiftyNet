# -*- coding: utf-8 -*-
"""
This module defines a general procedure for running applications
Example usage:
    app_driver = ApplicationDriver()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run_application()

system_param and input_data_param should be generated using:
niftynet.utilities.user_parameters_parser.run()
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

from niftynet.engine.application_factory import ApplicationFactory
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import GradientsCollector
from niftynet.engine.application_variables import NETORK_OUTPUT
from niftynet.engine.application_variables import OutputsCollector
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.application_variables import \
    global_vars_init_or_restore
from niftynet.io.misc_io import get_latest_subfolder
from niftynet.io.misc_io import touch_folder
from niftynet.layer.bn import BN_COLLECTION
from niftynet.utilities.util_common import set_cuda_device

FILE_PREFIX = 'model.ckpt'


class ApplicationDriver(object):
    """
    This class initialises an application by building a TF graph,
    and maintaining a session and coordinator. It controls the
    starting/stopping of an application. Applications should be
    implemented by inheriting niftynet.application.base_application
    to be compatible with this driver.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self.app = None
        self.graph = None
        self.saver = None

        self.is_training = True
        self.num_threads = 0
        self.num_gpus = 0

        self.model_dir = None
        self.summary_dir = None
        self.session_prefix = None
        self.max_checkpoints = 20
        self.save_every_n = 10
        self.tensorboard_every_n = 20
        self.initial_iter = 0
        self.final_iter = 0

        self._coord = None
        self._init_op = None
        self.outputs_collector = None
        self.gradients_collector = None

    def initialise_application(self, workflow_param, data_param):
        """
        This function receives all parameters from user config file,
        create an instance of application.
        :param workflow_param: a dictionary of user parameters,
        keys correspond to sections in the config file
        :param data_param: a dictionary of input image parameters,
        keys correspond to data properties to be used by image_reader
        :return:
        """
        try:
            system_param = workflow_param.get('SYSTEM', None)
            net_param = workflow_param.get('NETWORK', None)
            train_param = workflow_param.get('TRAINING', None)
            infer_param = workflow_param.get('INFERENCE', None)
            app_param = workflow_param.get('CUSTOM', None)
        except AttributeError:
            tf.logging.fatal('parameters should be dictionaries')
            raise

        self.is_training = (system_param.action == "train")
        # hardware-related parameters
        self.num_threads = max(system_param.num_threads, 1) \
            if self.is_training else 1
        self.num_gpus = system_param.num_gpus \
            if self.is_training else min(system_param.num_gpus, 1)
        set_cuda_device(system_param.cuda_devices)

        # set output folders
        self.model_dir = touch_folder(
            os.path.join(system_param.model_dir, 'models'))
        self.session_prefix = os.path.join(self.model_dir, FILE_PREFIX)

        if self.is_training:
            assert train_param, 'training parameters not specified'
            summary_root = os.path.join(self.model_dir, 'logs')
            self.summary_dir = get_latest_subfolder(
                summary_root, train_param.starting_iter == 0)

            # training iterations-related parameters
            self.initial_iter = train_param.starting_iter
            self.final_iter = train_param.max_iter
            self.save_every_n = train_param.save_every_n
            self.tensorboard_every_n = train_param.tensorboard_every_n
            self.max_checkpoints = train_param.max_checkpoints
            self.gradients_collector = GradientsCollector(
                n_devices=max(self.num_gpus, 1))
            action_param = train_param
        else:
            assert infer_param, 'inference parameters not specified'
            self.initial_iter = infer_param.inference_iter
            action_param = infer_param

        self.outputs_collector = OutputsCollector(
            n_devices=max(self.num_gpus, 1))

        # create an application instance
        assert app_param, 'application specific param. not specified'
        app_module = ApplicationDriver._create_app(app_param.name)
        self.app = app_module(net_param, action_param, self.is_training)
        # initialise data input
        self.app.initialise_dataset_loader(data_param, app_param)

    def run_application(self):
        """
        Initialise a TF graph, connect data sampler and network within
        the graph context, run training loops or inference loops.

        The training loop terminates when self.final_iter reached.
        The inference loop terminates when there is no more
        image sample to be processed from image reader.
        :return:
        """
        self.graph = self._create_graph()
        self.app.check_initialisations()
        config = ApplicationDriver._tf_config()
        with tf.Session(config=config, graph=self.graph) as session:
            # initialise network

            tf.logging.info('starting from iter %d', self.initial_iter)
            self._rand_init_or_restore_vars(session)

            # start samplers' threads
            tf.logging.info('Filling queues (this can take a few minutes)')
            self._coord = tf.train.Coordinator()
            for sampler in self.app.get_sampler():
                sampler.run_threads(session, self._coord, self.num_threads)

            start_time = time.time()
            loop_status = {}
            try:
                # iteratively run the graph
                if self.is_training:
                    loop_status['current_iter'] = self.initial_iter
                    self._training_loop(session, loop_status)
                else:
                    loop_status['all_saved_flag'] = False
                    self._inference_loop(session, loop_status)

            except KeyboardInterrupt:
                tf.logging.warning('User cancelled application')
            except tf.errors.OutOfRangeError:
                pass
            except RuntimeError:
                import sys
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(
                    exc_type, exc_value, exc_traceback, file=sys.stdout)
            finally:
                tf.logging.info('Cleaning up...')
                if self.is_training and loop_status.get('current_iter', None):
                    self._save_model(session, loop_status['current_iter'])
                elif loop_status.get('all_saved_flag', None):
                    if not loop_status['all_saved_flag']:
                        tf.logging.warning('stopped early, incomplete loops')

                tf.logging.info('stopping sampling threads')
                self.app.stop()
                tf.logging.info(
                    "%s stopped (time in second %.2f).",
                    type(self.app).__name__, (time.time() - start_time))

    # pylint: disable=not-context-manager
    def _create_graph(self):
        """
        tensorflow graph is only created within this function
        """
        graph = tf.Graph()
        main_device = self._device_string(0, is_worker=False)
        # start constructing the graph, handling training and inference cases
        with graph.as_default(), tf.device(main_device):
            # initialise sampler and network, these are connected in
            # the context of multiple gpus

            with tf.name_scope('Sampler'):
                self.app.initialise_sampler()
            self.app.initialise_network()

            # for data parallelism --
            #     defining and collecting variables from multiple devices
            bn_ops = None
            for gpu_id in range(0, max(self.num_gpus, 1)):
                worker_device = self._device_string(gpu_id, is_worker=True)
                scope_string = 'worker_{}'.format(gpu_id)
                with tf.name_scope(scope_string) as scope:
                    with tf.device(worker_device):
                        # setup network for each of the multiple devices
                        self.app.connect_data_and_network(
                            self.outputs_collector,
                            self.gradients_collector)
                        if self.is_training:
                            # batch norm statistics from the last device
                            bn_ops = tf.get_collection(BN_COLLECTION, scope)

            # assemble all training operations
            if self.is_training and self.gradients_collector:
                updates_op = []
                # batch normalisation moving averages operation
                if bn_ops:
                    updates_op.extend(bn_ops)
                # combine them with model parameter updating operation
                with tf.name_scope('ApplyGradients'):
                    with graph.control_dependencies(updates_op):
                        self.app.set_network_update_op(
                            self.gradients_collector.gradients)

            # initialisation operation
            with tf.name_scope('Initialization'):
                self._init_op = global_vars_init_or_restore()

            with tf.name_scope('MergedOutputs'):
                self.outputs_collector.finalise_output_op()
            # saving operation
            self.saver = tf.train.Saver(max_to_keep=self.max_checkpoints)

        # no more operation definitions after this point
        tf.Graph.finalize(graph)
        return graph

    def _rand_init_or_restore_vars(self, sess):
        """
        Randomly initialising all trainable variables defined in session,
        or loading checkpoint files as variable initialisations
        """
        if self.is_training and self.initial_iter == 0:
            sess.run(self._init_op)
            tf.logging.info('Parameters from random initialisations ...')
            return
        # check model's folder
        assert os.path.exists(self.model_dir), \
            "Model folder not found {}, please check" \
            "config parameter: model_dir".format(self.model_dir)

        # check model's file
        ckpt_state = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt_state is None:
            tf.logging.fatal(
                "%s/checkpoint not found, please check"
                "config parameter: model_dir", self.model_dir)
        if self.initial_iter > 0:
            checkpoint = '{}-{}'.format(self.session_prefix, self.initial_iter)
        else:
            try:
                checkpoint = ckpt_state.model_checkpoint_path
                assert checkpoint, 'checkpoint path not found ' \
                                   'in {}/checkpoints'.format(self.model_dir)
                self.initial_iter = int(checkpoint.rsplit('-')[-1])
                tf.logging.info('set initial_iter to %d based '
                                'on checkpoints', self.initial_iter)
            except (ValueError, AttributeError):
                tf.logging.fatal('failed to get iteration number'
                                 'from checkpoint path')
                raise
        # restore session
        tf.logging.info('Accessing %s ...', checkpoint)
        try:
            self.saver.restore(sess, checkpoint)
        except tf.errors.NotFoundError:
            tf.logging.fatal(
                'checkpoint %s not found or variables to restore do not '
                'match the current application graph', checkpoint)
            raise

    def _training_loop(self, sess, loop_status):
        """
        Training loop is running through the training_ops generator
        defined for each application (the application can specify
        training ops based on the current iteration number, this allows
        for complex optimisation schedules).

        At every iteration it also evaluates all variables returned by
        the output_collector.
        """
        writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
        # running through training_op from application
        for (iter_i, train_op) in \
                self.app.training_ops(self.initial_iter, self.final_iter):

            loop_status['current_iter'] = iter_i
            local_time = time.time()
            if self._coord.should_stop():
                break

            # variables to the graph
            vars_to_run = dict(train_op=train_op)
            vars_to_run[CONSOLE], vars_to_run[NETORK_OUTPUT] = \
                self.outputs_collector.variables(CONSOLE), \
                self.outputs_collector.variables(NETORK_OUTPUT)
            if self.tensorboard_every_n > 0 and \
                    (iter_i % self.tensorboard_every_n == 0):
                # adding tensorboard summary
                vars_to_run[TF_SUMMARIES] = \
                    self.outputs_collector.variables(collection=TF_SUMMARIES)

            # run all variables in one go
            graph_output = sess.run(vars_to_run)

            # process graph outputs
            self.app.interpret_output(graph_output[NETORK_OUTPUT])
            console_str = self._console_vars_to_str(graph_output[CONSOLE])
            summary = graph_output.get(TF_SUMMARIES, {})
            if summary:
                writer.add_summary(summary, iter_i)

            # save current model
            if (self.save_every_n > 0) and (iter_i % self.save_every_n == 0):
                self._save_model(sess, iter_i)
            tf.logging.info('iter %d, %s (%.3fs)',
                            iter_i, console_str, time.time() - local_time)

    def _inference_loop(self, sess, loop_status):
        """
        Runs all variables returned by outputs_collector,
        this loop stops when the return value of
        application.interpret_output is False.
        """
        loop_status['all_saved_flag'] = False
        while True:
            local_time = time.time()
            if self._coord.should_stop():
                break

            # build variables to run
            vars_to_run = dict()
            vars_to_run[NETORK_OUTPUT], vars_to_run[CONSOLE] = \
                self.outputs_collector.variables(NETORK_OUTPUT), \
                self.outputs_collector.variables(CONSOLE)

            # evaluate the graph variables
            graph_output = sess.run(vars_to_run)

            # process the graph outputs
            if not self.app.interpret_output(graph_output[NETORK_OUTPUT]):
                tf.logging.info('processed all batches.')
                loop_status['all_saved_flag'] = True
                break
            console_str = self._console_vars_to_str(graph_output[CONSOLE])
            tf.logging.info(
                '%s (%.3fs)', console_str, time.time() - local_time)

    def _save_model(self, session, iter_i):
        """
        save session parameters to the hard drive
        """
        if iter_i <= 0:
            return
        self.saver.save(sess=session,
                        save_path=self.session_prefix,
                        global_step=iter_i)
        tf.logging.info('iter %d saved: %s', iter_i, self.session_prefix)

    def _device_string(self, device_id=0, is_worker=True):
        """
        assigning CPU/GPU based on user specifications
        """
        # pylint: disable=no-name-in-module
        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()
        n_local_gpus = sum([x.device_type == 'GPU' for x in devices])
        if self.num_gpus <= 0:  # user specified no gpu at all
            return '/cpu:{}'.format(device_id)
        if self.is_training:
            # in training: use gpu only for workers whenever n_local_gpus
            device = 'gpu' if (is_worker and n_local_gpus > 0) else 'cpu'
            if device == 'gpu' and device_id >= n_local_gpus:
                tf.logging.fatal(
                    'trying to use gpu id %s, but only has %s GPU(s), '
                    'please set num_gpus to %s at most',
                    device_id, n_local_gpus, n_local_gpus)
                raise ValueError
            return '/{}:{}'.format(device, device_id)
        # in inference: use gpu for everything whenever n_local_gpus
        return '/gpu:0' if n_local_gpus > 0 else '/cpu:0'

    @staticmethod
    def _console_vars_to_str(console_dict):
        """
        Printing values of variable evaluations to command line output
        """
        if not console_dict:
            return ''
        console_str = ', '.join(
            '{}={}'.format(key, val) for (key, val) in console_dict.items())
        return console_str

    @staticmethod
    def _create_app(app_type_string):
        """
        Import the application module
        """
        return ApplicationFactory.create(app_type_string)

    @staticmethod
    def _tf_config():
        """
        tensorflow system configurations
        """
        config = tf.ConfigProto()
        config.log_device_placement = False
        config.allow_soft_placement = True
        return config
