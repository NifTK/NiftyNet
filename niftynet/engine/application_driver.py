# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import logging as log
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from niftynet.engine.graph_variables_collector import CONSOLE
from niftynet.engine.graph_variables_collector import GradientsCollector
from niftynet.engine.graph_variables_collector import OutputsCollector
from niftynet.engine.graph_variables_collector import TF_SUMMARIES
from niftynet.io.image_window import N_SPATIAL
from niftynet.utilities.misc_common import look_up_operations

FILE_PREFIX = 'model.ckpt'
CONSOLE_LOG_FORMAT = '%(levelname)s:niftynet: %(message)s'
FILE_LOG_FORMAT = '%(levelname)s:niftynet:%(asctime)s: %(message)s'
SUPPORTED_APP = {'net_segment.py',
                 'net_autoencoder.py',
                 'net_gan.py'}


class ApplicationFactory(object):
    @staticmethod
    def import_module(type_string):
        type_string = look_up_operations(type_string, SUPPORTED_APP)
        app_module = None
        if type_string == 'net_segment.py':
            from niftynet.application.segmentation_application import \
                SegmentationApplication
            app_module = SegmentationApplication
        elif type_string == 'net_autoencoder.py':
            from niftynet.application.autoencoder_application import \
                AutoencoderApplication
            app_module = AutoencoderApplication
        elif type_string == 'net_gan.py':
            from niftynet.application.gan_application import \
                GANApplication
            app_module = GANApplication
        else:
            raise NotImplementedError(
                'Unknown application name {}'.format(type_string))
        return app_module


class ApplicationDriver(object):
    def __init__(self):

        self.app = None
        self.graph = None
        self.saver = None

        self.is_training = True
        self.num_threads = 0
        self.num_gpus = 0

        self.model_dir = None
        self.session_dir = None
        self.summary_dir = None
        self.max_checkpoints = 20
        self.save_every_n = 10
        self.initial_iter = 0
        self.final_iter = 0

        self._init_op = None
        self.outputs_collector = None
        self.gradients_collector = None

    def initialise_application(self, system_param, data_param):
        app_param = system_param['APPLICATION']
        net_param = system_param['NETWORK']
        train_param = system_param['TRAINING']
        infer_param = system_param['INFERENCE']
        custom_param = system_param['CUSTOM']  # convert to a dictionary

        self.is_training = (app_param.action == "train")
        # hardware-related parameters
        self.num_threads = max(app_param.num_threads, 1) \
            if self.is_training else 1
        self.num_gpus = app_param.num_gpus \
            if self.is_training else min(app_param.num_gpus, 1)
        ApplicationDriver._set_cuda_device(app_param.cuda_devices)
        # set output folders
        self.model_dir = ApplicationDriver._touch_folder(app_param.model_dir)
        self.session_dir = os.path.join(self.model_dir, FILE_PREFIX)
        self.summary_dir = os.path.join(self.model_dir, 'logs')
        # set output logs to stdout and log file
        log_file_name = os.path.join(
            self.model_dir, '{}_{}'.format(app_param.action, 'log_console'))
        ApplicationDriver.set_logger(file_name=log_file_name)

        # model-related parameters
        self.initial_iter = train_param.starting_iter \
            if self.is_training else infer_param.inference_iter
        self.final_iter = train_param.max_iter
        self.save_every_n = train_param.save_every_n
        self.max_checkpoints = train_param.max_checkpoints

        self.outputs_collector = OutputsCollector(
            n_devices=max(self.num_gpus, 1))
        self.gradients_collector = GradientsCollector(
            n_devices=max(self.num_gpus, 1)) if self.is_training else None

        # create an application and assign user-specified parameters
        action_param = train_param if self.is_training else infer_param

        self.app = ApplicationDriver._create_app(custom_param.name)
        self.app.set_model_param(net_param, action_param, self.is_training)

        # initialise data input, and the tf graph
        self.app.initialise_dataset_loader(data_param, custom_param)
        self.graph = self._create_graph()

    def run_application(self):
        assert self.graph is not None, \
            "please call initialise_application first"
        config = ApplicationDriver._tf_config()
        with tf.Session(config=config, graph=self.graph) as session:
            # initialise network
            tf.logging.info('starting from iter {}'.format(self.initial_iter))
            self._randomly_init_or_restore_variables(session)

            # start samplers' threads
            self.coord = tf.train.Coordinator()
            samplers = self.app.get_sampler()
            tf.logging.info('Filling queues (this can take a few minutes)')
            for sampler in samplers:
                sampler.run_threads(session, self.coord, self.num_threads)

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
                tf.logging.warning('User cancelled training')
            except tf.errors.OutOfRangeError as e:
                pass
            except Exception:
                import sys, traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(
                    exc_type, exc_value, exc_traceback, file=sys.stdout)
            finally:
                if self.is_training and loop_status.get('current_iter', None):
                    self._save_model(session, loop_status['current_iter'])
                elif loop_status.get('all_saved_flag', None):
                    if not loop_status['all_saved_flag']:
                        tf.logging.warning('stopped early, incomplete loops')

                tf.logging.info('stopping sampling threads')
                self.app.stop()
                tf.logging.info("{} stopped (time in second {:.2f}).".format(
                    type(self.app).__name__, (time.time() - start_time)))

    def _create_graph(self):
        graph = tf.Graph()
        main_device = self._device_string(0, is_worker=False)
        # start constructing the graph, handling training and inference cases
        with graph.as_default(), tf.device(main_device):
            # initialise sampler and network, these are connected in
            # the context of multiple gpus
            self.app.initialise_sampler(is_training=self.is_training)
            self.app.initialise_network()

            # for data parallelism --
            #     defining and collecting variables from multiple devices
            for gpu_id in range(0, max(self.num_gpus, 1)):
                worker_device = self._device_string(gpu_id, is_worker=True)
                scope_string = 'worker_{}'.format(gpu_id)
                with tf.name_scope(scope_string) as scope:
                    with tf.device(worker_device):
                        # setup network for each of the multiple devices
                        self.app.connect_data_and_network(
                            self.outputs_collector,
                            self.gradients_collector)
                        # global batch norm statistics from the last device
                        bn_ops = tf.get_collection(
                            tf.GraphKeys.UPDATE_OPS, scope) \
                            if self.is_training else None

            # assemble all training operations
            if self.is_training:
                updates_op = []
                # model moving average operation
                mva_op = ApplicationDriver._model_moving_averaging_op()
                if not mva_op.type == "NoOp":
                    updates_op.extend(mva_op)
                # batch normalisation moving averages operation
                if bn_ops:
                    updates_op.extend(bn_ops)
                # combine them with model parameter updating operation
                if self.gradients_collector is not None:
                    with graph.control_dependencies(updates_op):
                        self.app.set_network_update_op(
                            self.gradients_collector.gradients)

            # initialisation operation
            self._init_op = tf.global_variables_initializer()

            self.outputs_collector.finalise_output_op()

            # saving operation
            self.saver = tf.train.Saver(max_to_keep=self.max_checkpoints)

        # no more operation definitions after this point
        tf.Graph.finalize(graph)
        return graph

    def _randomly_init_or_restore_variables(self, sess):
        if self.is_training and self.initial_iter == 0:
            sess.run(self._init_op)
            tf.logging.info('Parameters from random initialisations ...')
            return
        # check model's folder
        assert os.path.exists(self.model_dir), \
            "Model folder not found {}, please check" \
            "config parameter: model_dir".format(self.model_dir)

        # check model's file
        checkpoint = '{}-{}'.format(self.session_dir, self.initial_iter)
        assert tf.train.get_checkpoint_state(self.model_dir) is not None, \
            "Model file not found {}*, please check" \
            "config parameter: model_dir and *_iter".format(checkpoint)

        # restore session
        tf.logging.info('Accessing {} ...'.format(checkpoint))
        self.saver.restore(sess, checkpoint)
        return

    def _training_loop(self, sess, loop_status):
        writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
        # running through training_op from application
        for (iter_i, train_op) in \
                self.app.training_ops(self.initial_iter, self.final_iter):

            loop_status['current_iter'] = iter_i
            local_time = time.time()
            if self.coord.should_stop():
                break

            # prepare variables from the graph to run
            vars_to_run = dict(train_op=train_op)
            vars_to_run[CONSOLE] = \
                self.outputs_collector.variables(collection=CONSOLE)
            if iter_i % self.save_every_n == 0:
                # adding tensorboard summary
                vars_to_run[TF_SUMMARIES] = \
                    self.outputs_collector.variables(collection=TF_SUMMARIES)

            # run all variables in one go
            graph_output = sess.run(vars_to_run)
            self.app.interpret_output(graph_output, is_training=True)

            # if application specified summaries
            summary = graph_output.get(TF_SUMMARIES, {})
            if summary != {}:
                writer.add_summary(summary, iter_i)
                self._save_model(sess, iter_i)

            # print variables of the updated network
            console = graph_output.get(CONSOLE, {})
            console_str = ', '.join(
                '{}={}'.format(key, val) for (key, val) in console.items())
            tf.logging.info('iter {}, {} ({:.3f}s)'.format(
                iter_i, console_str, time.time() - local_time))

    def _inference_loop(self, sess, loop_status):
        loop_status['all_saved_flag'] = False
        while True:
            local_time = time.time()
            if self.coord.should_stop():
                break
            out = self.outputs_collector.variables()
            graph_output = sess.run(out)
            if not self.app.interpret_output(graph_output, is_training=False):
                tf.logging.info('processed all batches.')
                loop_status['all_saved_flag'] = True
                break
            tf.logging.info('{:.3f}s'.format(time.time() - local_time))

    def _save_model(self, session, iter_i):
        if iter_i <= 0:
            return
        self.saver.save(sess=session,
                        save_path=self.session_dir,
                        global_step=iter_i)
        tf.logging.info('iter {} saved: {}'.format(iter_i, self.session_dir))

    def _device_string(self, id=0, is_worker=True):

        devices = device_lib.list_local_devices()
        has_local_gpu = any([x.device_type == 'GPU' for x in devices])
        if self.num_gpus <= 0:  # user specified no gpu at all
            return '/cpu:{}'.format(id)
        # in training: use gpu only for workers whenever has_local_gpu
        # in inference: use gpu for everything whenever has_local_gpu
        if self.is_training:
            device = 'gpu' if (is_worker and has_local_gpu) else 'cpu'
            return '/{}:{}'.format(device, id)
        else:
            return '/gpu:0' if has_local_gpu else '/cpu:0'

    @staticmethod
    def _touch_folder(model_dir):
        model_dir = os.path.join(model_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        absolute_dir = os.path.abspath(model_dir)
        tf.logging.info('accessing output folder: {}'.format(absolute_dir))
        return absolute_dir

    @staticmethod
    def _set_cuda_device(cuda_devices):
        # TODO: refactor this OS-dependent function
        if not (cuda_devices == '""'):
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
            tf.logging.info(
                "set CUDA_VISIBLE_DEVICES to {}".format(cuda_devices))
        else:
            # using Tensorflow default choice
            pass

    @staticmethod
    def _model_moving_averaging_op(decay=0.9):
        variable_averages = tf.train.ExponentialMovingAverage(decay)
        trainables = tf.trainable_variables()
        return variable_averages.apply(var_list=trainables)

    @staticmethod
    def _create_app(app_type_string):
        _app_module = ApplicationFactory.import_module(app_type_string)
        return _app_module()

    @staticmethod
    def _tf_config():
        config = tf.ConfigProto()
        config.log_device_placement = False
        config.allow_soft_placement = True
        return config

    @staticmethod
    def set_logger(file_name=None):
        tf.logging._logger.handlers = []
        tf.logging._logger = log.getLogger('tensorflow')
        tf.logging.set_verbosity(tf.logging.INFO)

        f = log.Formatter(CONSOLE_LOG_FORMAT)
        std_handler = log.StreamHandler(sys.stdout)
        std_handler.setFormatter(f)
        tf.logging._logger.addHandler(std_handler)

        if file_name is not None:
            f = log.Formatter(FILE_LOG_FORMAT)
            file_handler = log.FileHandler(file_name)
            file_handler.setFormatter(f)
            tf.logging._logger.addHandler(file_handler)

