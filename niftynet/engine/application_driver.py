# -*- coding: utf-8 -*-
"""
This module defines a general procedure for running applications.

Example usage::
    app_driver = ApplicationDriver()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run_application()

``system_param`` and ``input_data_param`` should be generated using:
``niftynet.utilities.user_parameters_parser.run()``
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import time

import tensorflow as tf

from niftynet.engine.application_factory import \
    ApplicationFactory, EventHandlerFactory
from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.application_variables import \
    GradientsCollector, OutputsCollector
from niftynet.engine.signal import TRAIN, VALID, INFER, \
    ITER_STARTED, ITER_FINISHED, SESS_STARTED, SESS_FINISHED
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.io.misc_io import get_latest_subfolder, touch_folder
from niftynet.layer.bn import BN_COLLECTION
from niftynet.utilities.util_common import set_cuda_device, traverse_nested


# pylint: disable=too-many-instance-attributes
class ApplicationDriver(object):
    """
    This class initialises an application by building a TF graph,
    and maintaining a session and coordinator. It controls the
    starting/stopping of an application. Applications should be
    implemented by inheriting ``niftynet.application.base_application``
    to be compatible with this driver.
    """

    def __init__(self):
        self.app = None
        self.graph = tf.Graph()

        self.is_training = True
        self.num_threads = 0
        self.num_gpus = 0

        self.model_dir = None
        self.summary_dir = None
        self.max_checkpoints = 2

        self.initial_iter = 0
        self.final_iter = 0

        self.save_every_n = 0
        self.tensorboard_every_n = -1
        self.validation_every_n = -1
        self.validation_max_iter = 1

        self._coord = tf.train.Coordinator()
        self._data_partitioner = None
        self.outputs_collector = None
        self.gradients_collector = None

        self.event_handler_names = [
            'niftynet.engine.event_gradient.ApplyGradients',
            'niftynet.engine.event_console.ConsoleLogger',
            'niftynet.engine.event_tensorboard.TensorBoardLogger',
            'niftynet.engine.event_checkpoint.ModelSaver',
            'niftynet.engine.event_network_output.OutputInterpreter']
        self._event_handlers = []

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

        assert os.path.exists(system_param.model_dir), \
            'Model folder not exists {}'.format(system_param.model_dir)
        self.is_training = TRAIN.startswith(system_param.action.lower())
        # hardware-related parameters
        self.num_threads = max(system_param.num_threads, 1) \
            if self.is_training else 1
        self.num_gpus = system_param.num_gpus \
            if self.is_training else min(system_param.num_gpus, 1)
        set_cuda_device(system_param.cuda_devices)

        # set output TF model folders
        self.model_dir = touch_folder(
            os.path.join(system_param.model_dir, 'models'))

        # set training params.
        if self.is_training:
            assert train_param, 'training parameters not specified'
            summary_root = os.path.join(system_param.model_dir, 'logs')
            self.summary_dir = get_latest_subfolder(
                summary_root, create_new=train_param.starting_iter == 0)

            self.initial_iter = train_param.starting_iter
            self.final_iter = max(train_param.max_iter, self.initial_iter)
            self.save_every_n = train_param.save_every_n
            self.tensorboard_every_n = train_param.tensorboard_every_n
            self.max_checkpoints = \
                max(train_param.max_checkpoints, self.max_checkpoints)
            self.gradients_collector = GradientsCollector(
                n_devices=max(self.num_gpus, 1))
            self.validation_every_n = train_param.validation_every_n
            if self.validation_every_n > 0:
                self.validation_max_iter = max(self.validation_max_iter,
                                               train_param.validation_max_iter)
            action_param = train_param
        else:  # set inference params.
            assert infer_param, 'inference parameters not specified'
            self.initial_iter = infer_param.inference_iter
            action_param = infer_param

        self.outputs_collector = OutputsCollector(
            n_devices=max(self.num_gpus, 1))

        # create an application instance
        assert app_param, 'application specific param. not specified'
        app_module = ApplicationDriver._create_app(app_param.name)
        self.app = app_module(net_param, action_param, system_param.action)

        # initialise data input
        self._data_partitioner = ImageSetsPartitioner()
        # clear the cached file lists
        self._data_partitioner.reset()
        do_new_partition = \
            self.is_training and self.initial_iter == 0 and \
            (not os.path.isfile(system_param.dataset_split_file)) and \
            (train_param.exclude_fraction_for_validation > 0 or
             train_param.exclude_fraction_for_inference > 0)
        data_fractions = None
        if do_new_partition:
            assert train_param.exclude_fraction_for_validation > 0 or \
                self.validation_every_n <= 0, \
                'validation_every_n is set to {}, ' \
                'but train/validation splitting not available,\nplease ' \
                'check "exclude_fraction_for_validation" in the config ' \
                'file (current config value: {}).'.format(
                    self.validation_every_n,
                    train_param.exclude_fraction_for_validation)
            data_fractions = (train_param.exclude_fraction_for_validation,
                              train_param.exclude_fraction_for_inference)
        if data_param:
            self._data_partitioner.initialise(
                data_param=data_param,
                new_partition=do_new_partition,
                ratios=data_fractions,
                data_split_file=system_param.dataset_split_file)
        if data_param and self.is_training and self.validation_every_n > 0:
            assert self._data_partitioner.has_validation, \
                'validation_every_n is set to {}, ' \
                'but train/validation splitting not available.\nPlease ' \
                'check dataset partition list {} ' \
                '(remove file to generate a new dataset partition). ' \
                'Or set validation_every_n to -1.'.format(
                    self.validation_every_n, system_param.dataset_split_file)

        # initialise readers
        self.app.initialise_dataset_loader(
            data_param, app_param, self._data_partitioner)

        # pylint: disable=not-context-manager
        with self.graph.as_default(), tf.name_scope('Sampler'):
            self.app.initialise_sampler()

    def load_event_handlers(self, names):
        """
        Import event handler modules and create a list of handler instances.
        The event handler instances will be stored with this engine.

        :param names: strings of event handlers
        :return:
        """
        self._event_handlers = []
        for name in set(names):
            the_event_class = EventHandlerFactory.create(name)
            # initialise all registered event handler classes
            self._event_handlers.append(the_event_class(**vars(self)))

    def run_application(self):
        """
        Initialise a TF graph, connect data sampler and network within
        the graph context, run training loops or inference loops.

        The training loop terminates when ``self.final_iter`` reached.
        The inference loop terminates when there is no more
        image sample to be processed from image reader.

        :return:
        """
        config = ApplicationDriver._tf_config()
        with tf.Session(config=config, graph=self.graph) as session:

            # start samplers' threads
            self._run_sampler_threads(session=session)
            self.graph = self._create_graph(self.graph)

            # check app variables initialised and ready for starts
            self.app.check_initialisations()

            # make the list of initialised event handler instances.
            self.load_event_handlers(self.event_handler_names)

            start_time = time.time()
            loop_status = {}

            try:
                # broadcasting event of session started
                SESS_STARTED.send(self.app, iter_msg=None)

                if self.is_training:
                    loop_status['current_iter'] = self.initial_iter
                    iterator = train_iter_generator(
                        self.initial_iter,
                        self.final_iter,
                        self.validation_every_n,
                        self.validation_max_iter)
                else:
                    loop_status['all_saved_flag'] = False
                    iterator = infer_iter_generator()

                # iteratively run the graph
                self._loop(iterator, session, loop_status)

            except KeyboardInterrupt:
                tf.logging.warning('User cancelled application')
            except tf.errors.OutOfRangeError:
                if loop_status.get('all_saved_flag', None) is not None:
                    # reached the end of inference Dataset
                    loop_status['all_saved_flag'] = True
            except RuntimeError:
                import sys
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(
                    exc_type, exc_value, exc_traceback, file=sys.stdout)
            finally:
                # broadcasting event of session finished
                iter_msg = IterationMessage()
                iter_msg.current_iter = loop_status.get('current_iter', -1)
                SESS_FINISHED.send(self.app, iter_msg=iter_msg)

                tf.logging.info('Cleaning up...')
                if not self.is_training and \
                        not loop_status.get('all_saved_flag', None):
                    tf.logging.warning('stopped early, incomplete loops')

                tf.logging.info('stopping sampling threads')
                self.app.stop()
                tf.logging.info(
                    "%s stopped (time in second %.2f).",
                    type(self.app).__name__, (time.time() - start_time))

    # pylint: disable=not-context-manager
    def _create_graph(self, graph=tf.Graph()):
        """
        TensorFlow graph is only created within this function.
        """
        assert isinstance(graph, tf.Graph)
        main_device = self._device_string(0, is_worker=False)
        # start constructing the graph, handling training and inference cases
        with graph.as_default(), tf.device(main_device):

            # initialise network, these are connected in
            # the context of multiple gpus
            self.app.initialise_network()
            self.app.add_validation_flag()

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
                        self.app.set_network_gradient_op(
                            self.gradients_collector.gradients)

            # # initialisation operation
            # with tf.name_scope('Initialization'):
            #     self.init_op = global_vars_init_or_restore()

            with tf.name_scope('MergedOutputs'):
                self.outputs_collector.finalise_output_op()

        # tf.Graph.finalize(graph)
        return graph

    def _loop(self, iteration_generator, sess=None, loop_status=None):
        """
        This loop stops when any of the condition satisfied:
            1. no more element from the ``iteration_generator``;
            2. ``application.interpret_output`` returns False;
            3. any exception raised.

        :param iteration_generator:
            iteratively generating ``engine.IterationMessage`` instances
        :param sess: Tensorflow session
        :param loop_status: dictionary used to capture the loop status,
            useful when the loop exited in an unexpected manner.
        :return:
        """
        if not sess:
            return

        loop_status = loop_status or {}
        for iter_msg in iteration_generator:
            if self._coord.should_stop():
                break
            loop_status['current_iter'] = iter_msg.current_iter

            # broadcasting event of starting an iteration
            ITER_STARTED.send(self.app, iter_msg=iter_msg)

            # ``iter_msg.ops_to_run`` are populated with the ops to run in
            #  each iteration, fed into ``session.run()`` and then
            # passed to the app (and observers) for interpretation.
            graph_output = sess.run(
                iter_msg.ops_to_run, feed_dict=iter_msg.data_feed_dict)
            iter_msg.current_iter_output = graph_output

            # broadcasting event of finishing an iteration
            ITER_FINISHED.send(self.app, iter_msg=iter_msg)

            # Checking stopping conditions
            if iter_msg.should_stop:
                loop_status['all_saved_flag'] = True
                tf.logging.info('Stopping message from event handler: %s.',
                                iter_msg.should_stop)
                break

    def _run_sampler_threads(self, session=None):
        """
        Get samplers from application and try to run sampler threads.

        Note: Overriding app.get_sampler() method by returning None to bypass
        this step.

        :param session: TF session used for fill
            tf.placeholders with sampled data
        :return:
        """
        if session is None:
            return
        if self._coord is None:
            return
        if self.num_threads <= 0:
            return
        try:
            samplers = self.app.get_sampler()
            for sampler in traverse_nested(samplers):
                if sampler is None:
                    continue
                sampler.run_threads(session, self._coord, self.num_threads)
            tf.logging.info('Filling queues (this can take a few minutes)')
        except (NameError, TypeError, AttributeError, IndexError):
            tf.logging.fatal(
                "samplers not running, pop_batch_op operations "
                "are blocked.")
            raise

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
                tf.logging.warning(
                    'trying to use gpu id %s, but only has %s GPU(s), '
                    'please set num_gpus to %s at most',
                    device_id, n_local_gpus, n_local_gpus)
                # raise ValueError
            return '/{}:{}'.format(device, device_id)
        # in inference: use gpu for everything whenever n_local_gpus
        return '/gpu:0' if n_local_gpus > 0 else '/cpu:0'

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


def iter_generator(count_generator, phase):
    """
    Generate a numbered sequence of IterationMessage objects
    with phase-appropriate signals.
    count_generator is an iterable object yielding iteration numbers
    phase is one of TRAIN, VALID or INFER
    """
    for iter_i in count_generator:
        iter_msg = IterationMessage()
        iter_msg.current_iter, iter_msg.phase = iter_i, phase
        yield iter_msg


def infer_iter_generator():
    """
    This generator yields infinite number of infer iterations.

    :return: iteration message instances
    """
    infer_iterations = iter_generator(itertools.count(), INFER)
    for infer_iter_msg in infer_iterations:
        yield infer_iter_msg


def train_iter_generator(initial_iter=0,
                         max_iter=0,
                         validation_every_n=0,
                         validation_max_iter=0):
    """
    This generator yields a sequence of interleaved training and validation
    iterations.

    :param initial_iter: starting iteration of the training sequence
    :param max_iter: ending iteration of the training sequence
    :param validation_every_n: validation at every n training
    :param validation_max_iter: number of validation iterations
    :return: iteration message instances
    """
    train_iterations = iter_generator(
        range(initial_iter + 1, max_iter + 1), TRAIN)
    for train_iter_msg in train_iterations:
        yield train_iter_msg
        current_iter = train_iter_msg.current_iter
        if current_iter > 0 and validation_every_n > 0 and \
                current_iter % validation_every_n == 0:
            valid_iterations = iter_generator(
                [current_iter] * validation_max_iter, VALID)
            for valid_iter_msg in valid_iterations:
                yield valid_iter_msg
