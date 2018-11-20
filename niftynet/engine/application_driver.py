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

import os
import time

import tensorflow as tf

from niftynet.engine.application_factory import \
    ApplicationFactory, EventHandlerFactory, IteratorFactory
from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.application_variables import \
    GradientsCollector, OutputsCollector
from niftynet.engine.signal import TRAIN, \
    ITER_STARTED, ITER_FINISHED, GRAPH_CREATED, SESS_FINISHED, SESS_STARTED
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.io.misc_io import infer_latest_model_file
from niftynet.utilities.user_parameters_default import \
    DEFAULT_EVENT_HANDLERS, DEFAULT_ITERATION_GENERATOR
from niftynet.utilities.util_common import \
    set_cuda_device, tf_config, device_string
from niftynet.utilities.util_common import traverse_nested


# pylint: disable=too-many-instance-attributes
class ApplicationDriver(object):
    """
    This class initialises an application by building a TF graph,
    and maintaining a session. It controls the
    starting/stopping of an application. Applications should be
    implemented by inheriting ``niftynet.application.base_application``
    to be compatible with this driver.
    """

    def __init__(self):
        self.app = None

        self.is_training_action = True
        self.num_threads = 0
        self.num_gpus = 0
        self.model_dir = None

        self.max_checkpoints = 2
        self.save_every_n = 0
        self.tensorboard_every_n = -1
        self.vars_to_restore = ''

        self.initial_iter = 0
        self.final_iter = 0
        self.validation_every_n = -1
        self.validation_max_iter = 1

        self.data_partitioner = ImageSetsPartitioner()

        self._event_handlers = None
        self._generator = None

    def initialise_application(self, workflow_param, data_param=None):
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
        self.model_dir = system_param.model_dir

        self.is_training_action = TRAIN.startswith(system_param.action.lower())
        # hardware-related parameters
        self.num_threads = max(system_param.num_threads, 1) \
            if self.is_training_action else 1
        self.num_gpus = system_param.num_gpus \
            if self.is_training_action else min(system_param.num_gpus, 1)
        set_cuda_device(system_param.cuda_devices)

        # set training params.
        if self.is_training_action:
            assert train_param, 'training parameters not specified'
            self.initial_iter = train_param.starting_iter
            self.final_iter = max(train_param.max_iter, self.initial_iter)
            self.save_every_n = train_param.save_every_n
            self.tensorboard_every_n = train_param.tensorboard_every_n
            self.max_checkpoints = max(self.max_checkpoints,
                                       train_param.max_checkpoints)
            self.validation_every_n = train_param.validation_every_n
            self.vars_to_restore = train_param.vars_to_restore \
                if hasattr(train_param, 'vars_to_restore') else ''
            if self.validation_every_n > 0:
                self.validation_max_iter = max(self.validation_max_iter,
                                               train_param.validation_max_iter)
            action_param = train_param
        else:  # set inference params.
            assert infer_param, 'inference parameters not specified'
            self.initial_iter = infer_param.inference_iter
            action_param = infer_param

        # infer the initial iteration from model files
        if self.initial_iter < 0:
            self.initial_iter = infer_latest_model_file(
                os.path.join(self.model_dir, 'models'))

        # create an application instance
        assert app_param, 'application specific param. not specified'
        app_module = ApplicationFactory.create(app_param.name)
        self.app = app_module(net_param, action_param, system_param.action)

        # clear the cached file lists
        self.data_partitioner.reset()
        if data_param:
            do_new_partition = \
                self.is_training_action and \
                (not os.path.isfile(system_param.dataset_split_file)) and \
                (train_param.exclude_fraction_for_validation > 0 or
                 train_param.exclude_fraction_for_inference > 0)
            data_fractions = (train_param.exclude_fraction_for_validation,
                              train_param.exclude_fraction_for_inference) \
                if do_new_partition else None

            self.data_partitioner.initialise(
                data_param=data_param,
                new_partition=do_new_partition,
                ratios=data_fractions,
                data_split_file=system_param.dataset_split_file)
            assert self.data_partitioner.has_validation or \
                self.validation_every_n <= 0, \
                'validation_every_n is set to {}, ' \
                'but train/validation splitting not available.\nPlease ' \
                'check dataset partition list {} ' \
                '(remove file to generate a new dataset partition), ' \
                'check "exclude_fraction_for_validation" ' \
                '(current config value: {}).\nAlternatively, ' \
                'set "validation_every_n" to -1.'.format(
                    self.validation_every_n,
                    system_param.dataset_split_file,
                    train_param.exclude_fraction_for_validation)

        # initialise readers
        self.app.initialise_dataset_loader(
            data_param, app_param, self.data_partitioner)

        # make the list of initialised event handler instances.
        self.load_event_handlers(
            system_param.event_handler or DEFAULT_EVENT_HANDLERS)
        self._generator = IteratorFactory.create(
            system_param.iteration_generator or DEFAULT_ITERATION_GENERATOR)

    def run(self, application, graph=None):
        """
        Initialise a TF graph, connect data sampler and network within
        the graph context, run training loops or inference loops.

        :param application: a niftynet application
        :param graph: default base graph to run the application
        :return:
        """
        if graph is None:
            graph = ApplicationDriver.create_graph(
                application=application,
                num_gpus=self.num_gpus,
                num_threads=self.num_threads,
                is_training_action=self.is_training_action)

        start_time = time.time()
        loop_status = {'current_iter': self.initial_iter, 'normal_exit': False}

        with tf.Session(config=tf_config(), graph=graph):
            try:
                # broadcasting event of session started
                SESS_STARTED.send(application, iter_msg=None)

                # create a iteration message generator and
                # iteratively run the graph (the main engine loop)
                iteration_messages = self._generator(**vars(self))()
                ApplicationDriver.loop(
                    application=application,
                    iteration_messages=iteration_messages,
                    loop_status=loop_status)

            except KeyboardInterrupt:
                tf.logging.warning('User cancelled application')
            except (tf.errors.OutOfRangeError, EOFError):
                if not loop_status.get('normal_exit', False):
                    # reached the end of inference Dataset
                    loop_status['normal_exit'] = True
            except RuntimeError:
                import sys
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(
                    exc_type, exc_value, exc_traceback, file=sys.stdout)
            finally:
                tf.logging.info('cleaning up...')
                # broadcasting session finished event
                iter_msg = IterationMessage()
                iter_msg.current_iter = loop_status.get('current_iter', -1)
                SESS_FINISHED.send(application, iter_msg=iter_msg)

        application.stop()
        if not loop_status.get('normal_exit', False):
            # loop didn't finish normally
            tf.logging.warning('stopped early, incomplete iterations.')
        tf.logging.info(
            "%s stopped (time in second %.2f).",
            type(application).__name__, (time.time() - start_time))

    # pylint: disable=not-context-manager
    @staticmethod
    def create_graph(
            application, num_gpus=1, num_threads=1, is_training_action=False):
        """
        Create a TF graph based on self.app properties
        and engine parameters.

        :return:
        """
        graph = tf.Graph()
        main_device = device_string(num_gpus, 0, False, is_training_action)
        outputs_collector = OutputsCollector(n_devices=max(num_gpus, 1))
        gradients_collector = GradientsCollector(n_devices=max(num_gpus, 1))
        # start constructing the graph, handling training and inference cases
        with graph.as_default(), tf.device(main_device):
            # initialise sampler
            with tf.name_scope('Sampler'):
                application.initialise_sampler()
                for sampler in traverse_nested(application.get_sampler()):
                    sampler.set_num_threads(num_threads)

            # initialise network, these are connected in
            # the context of multiple gpus
            application.initialise_network()
            application.add_validation_flag()

            # for data parallelism --
            #     defining and collecting variables from multiple devices
            for gpu_id in range(0, max(num_gpus, 1)):
                worker_device = device_string(
                    num_gpus, gpu_id, True, is_training_action)
                scope_string = 'worker_{}'.format(gpu_id)
                with tf.name_scope(scope_string), tf.device(worker_device):
                    # setup network for each of the multiple devices
                    application.connect_data_and_network(
                        outputs_collector, gradients_collector)
            with tf.name_scope('MergeOutputs'):
                outputs_collector.finalise_output_op()
            application.outputs_collector = outputs_collector
            application.gradients_collector = gradients_collector
            GRAPH_CREATED.send(application, iter_msg=None)
        return graph

    def load_event_handlers(self, names):
        """
        Import event handler modules and create a list of handler instances.
        The event handler instances will be stored with this engine.

        :param names: strings of event handlers
        :return:
        """
        if not names:
            return
        if self._event_handlers:
            # disconnect all handlers (assuming always weak connection)
            for handler in list(self._event_handlers):
                del self._event_handlers[handler]
        self._event_handlers = {}
        for name in set(names):
            the_event_class = EventHandlerFactory.create(name)
            # initialise all registered event handler classes
            engine_config_dict = vars(self)
            key = '{}'.format(the_event_class)
            self._event_handlers[key] = the_event_class(**engine_config_dict)

    @staticmethod
    def loop(application,
             iteration_messages=(),
             loop_status=None):
        """
        Running ``loop_step`` with ``IterationMessage`` instances
        generated by ``iteration_generator``.

        This loop stops when any of the condition satisfied:
            1. no more element from the ``iteration_generator``;
            2. ``application.interpret_output`` returns False;
            3. any exception raised.

        Broadcasting SESS_* signals at the beginning and end of this method.

        This function should be used in a context of
        ``tf.Session`` or ``session.as_default()``.

        :param application: a niftynet.application instance, application
            will provides ``tensors`` to be fetched by ``tf.session.run()``.
        :param iteration_messages:
            a generator of ``engine.IterationMessage`` instances
        :param loop_status: optional dictionary used to capture the loop status,
            useful when the loop exited in an unexpected manner.
        :return:
        """
        loop_status = loop_status or {}
        for iter_msg in iteration_messages:
            loop_status['current_iter'] = iter_msg.current_iter

            # run an iteration
            ApplicationDriver.loop_step(application, iter_msg)

            # Checking stopping conditions
            if iter_msg.should_stop:
                tf.logging.info('stopping -- event handler: %s.',
                                iter_msg.should_stop)
                break
        # loop finished without any exception
        loop_status['normal_exit'] = True

    @staticmethod
    def loop_step(application, iteration_message):
        """
        Calling ``tf.session.run`` with parameters encapsulated in
        iteration message as an iteration.
        Broadcasting ITER_* events before and afterward.

        :param application:
        :param iteration_message: an ``engine.IterationMessage`` instances
        :return:
        """
        # broadcasting event of starting an iteration
        ITER_STARTED.send(application, iter_msg=iteration_message)

        # ``iter_msg.ops_to_run`` are populated with the ops to run in
        # each iteration, fed into ``session.run()`` and then
        # passed to the application (and observers) for interpretation.
        sess = tf.get_default_session()
        assert sess, 'method should be called within a TF session context.'

        iteration_message.current_iter_output = sess.run(
            iteration_message.ops_to_run,
            feed_dict=iteration_message.data_feed_dict)

        # broadcasting event of finishing an iteration
        ITER_FINISHED.send(application, iter_msg=iteration_message)
