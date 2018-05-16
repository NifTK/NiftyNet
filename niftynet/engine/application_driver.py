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
import itertools

import tensorflow as tf
from blinker import signal

from niftynet.engine.application_factory import ApplicationFactory
from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.engine.application_variables import \
    GradientsCollector, OutputsCollector, global_vars_init_or_restore
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.io.image_sets_partitioner import TRAIN, VALID, INFER
from niftynet.io.misc_io import get_latest_subfolder, touch_folder
from niftynet.layer.bn import BN_COLLECTION
from niftynet.utilities.util_common import set_cuda_device, traverse_nested

FILE_PREFIX = 'model.ckpt'

# pylint: disable=too-many-instance-attributes
class ApplicationDriver(object):
    """
    This class initialises an application by building a TF graph,
    and maintaining a session and coordinator. It controls the
    starting/stopping of an application. Applications should be
    implemented by inheriting ``niftynet.application.base_application``
    to be compatible with this driver.
    """

    # pylint: disable=too-many-instance-attributes

    pre_train_iter = signal('pre_train_iter')
    post_train_iter = signal('post_train_iter')
    pre_validation_iter = signal('pre_validation_iter')
    post_validation_iter = signal('post_validation_iter')
    pre_infer_iter = signal('pre_infer_iter')
    post_infer_iter = signal('post_infer_iter')
    post_training = signal('post_training')

    def __init__(self):
        self.app = None
        self.graph = tf.Graph()

        self.saver = None

        self.is_training = True
        self.num_threads = 0
        self.num_gpus = 0

        self.model_dir = None
        self.summary_dir = None
        self.session_prefix = None
        self.max_checkpoints = 2
        self.save_every_n = 0
        self.tensorboard_every_n = -1

        self.validation_every_n = -1
        self.validation_max_iter = 1

        self.initial_iter = 0
        self.final_iter = 0

        self._coord = tf.train.Coordinator()
        self._init_op = None
        self._data_partitioner = None
        self.outputs_collector = None
        self.gradients_collector = None

        self.console = None
        self.tensorboard = None
        self.model_saver = None

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
        self.is_training = (system_param.action == "train")
        # hardware-related parameters
        self.num_threads = max(system_param.num_threads, 1) \
            if self.is_training else 1
        self.num_gpus = system_param.num_gpus \
            if self.is_training else min(system_param.num_gpus, 1)
        set_cuda_device(system_param.cuda_devices)

        # set output TF model folders
        self.model_dir = touch_folder(
            os.path.join(system_param.model_dir, 'models'))
        self.session_prefix = os.path.join(self.model_dir, FILE_PREFIX)

        # set training params.
        if self.is_training:
            assert train_param, 'training parameters not specified'
            summary_root = os.path.join(system_param.model_dir, 'logs')
            self.summary_dir = get_latest_subfolder(
                summary_root,
                create_new=train_param.starting_iter == 0)

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
        else: # set inference params.
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
        data_partitioner = ImageSetsPartitioner()
        # clear the cached file lists
        data_partitioner.reset()
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
            data_partitioner.initialise(
                data_param=data_param,
                new_partition=do_new_partition,
                ratios=data_fractions,
                data_split_file=system_param.dataset_split_file)

        if data_param and self.is_training and self.validation_every_n > 0:
            assert data_partitioner.has_validation, \
                'validation_every_n is set to {}, ' \
                'but train/validation splitting not available.\nPlease ' \
                'check dataset partition list {} ' \
                '(remove file to generate a new dataset partition). ' \
                'Or set validation_every_n to -1.'.format(
                    self.validation_every_n, system_param.dataset_split_file)

        # initialise readers
        self.app.initialise_dataset_loader(
            data_param, app_param, data_partitioner)

        self._data_partitioner = data_partitioner

        # pylint: disable=not-context-manager
        with self.graph.as_default(), tf.name_scope('Sampler'):
            self.app.initialise_sampler()

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

            # initialise network trainable parameters
            self._rand_init_or_restore_vars(session)

            start_time = time.time()
            loop_status = {}
            try:
                # iteratively run the graph
                if self.is_training:
                    self.model_saver = ModelSaver(session, self.saver,
                                                  self.save_every_n,
                                                  self.session_prefix)
                    loop_status['current_iter'] = self.initial_iter
                    self._training_loop(session, loop_status)
                else:
                    loop_status['all_saved_flag'] = False
                    self._inference_loop(session, loop_status)

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
                tf.logging.info('Cleaning up...')
                if self.is_training:
                    # saving model at the last iteration
                    iter_msg = IterationMessage()
                    iter_msg.current_iter = loop_status.get('current_iter', -1)
                    self.post_training.send(iter_msg)
                elif not loop_status.get('all_saved_flag', None):
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

            # initialisation operation
            with tf.name_scope('Initialization'):
                self._init_op = global_vars_init_or_restore()

            with tf.name_scope('MergedOutputs'):
                self.outputs_collector.finalise_output_op()
            # saving operation
            self.saver = tf.train.Saver(max_to_keep=self.max_checkpoints,
                                        save_relative_paths=True)

        # no more operation definitions after this point
        tf.Graph.finalize(graph)
        return graph

    def _rand_init_or_restore_vars(self, sess):
        """
        Randomly initialising all trainable variables defined in session,
        or loading checkpoint files as variable initialisations.
        """
        tf.logging.info('starting from iter %d', self.initial_iter)
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
            tf.logging.warning(
                "%s/checkpoint not found, please check "
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
                tf.logging.fatal(
                    'failed to get iteration number '
                    'from checkpoint path, please set '
                    'inference_iter or starting_iter to a positive integer')
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

    def interleaved_iteration_generator(self):
        """ This generator yields a sequence of training and validation
        iterations """
        train_iters = iter_generator(range(self.initial_iter + 1,
                                           self.final_iter + 1), TRAIN)
        for train_iter_msg in train_iters:
            self.app.set_iteration_update(train_iter_msg)
            yield train_iter_msg
            if train_iter_msg.current_iter > 0 and\
                  self.validation_every_n > 0 and \
                  (train_iter_msg.current_iter % self.validation_every_n == 0):
                val_iters = [train_iter_msg.current_iter]
                val_iters = val_iters * self.validation_max_iter
                valid_iters = iter_generator(val_iters, VALID)
                for valid_iter_msg in valid_iters:
                    self.app.set_iteration_update(valid_iter_msg)
                    yield valid_iter_msg

    def _loop(self, iteration_generator, sess, loop_status):
        for iter_msg in iteration_generator:
            if self._coord.should_stop():
                break
            if iter_msg.should_stop:
                break
            loop_status['current_iter'] = iter_msg.current_iter
            iter_msg.pre_iter.send(iter_msg)

            iter_msg.ops_to_run[NETWORK_OUTPUT] = \
                self.outputs_collector.variables(NETWORK_OUTPUT)
            graph_output = sess.run(iter_msg.ops_to_run,
                                    feed_dict=iter_msg.data_feed_dict)
            iter_msg.current_iter_output = graph_output
            iter_msg.status = self.app.interpret_output(
                iter_msg.current_iter_output[NETWORK_OUTPUT])

            iter_msg.post_iter.send(iter_msg)

            if iter_msg.should_stop:
                break

    def _training_loop(self, sess, loop_status):
        """
        The training loop iterates through training (and validation) iterations
        Each iteration is represented as an ``IterationMessage`` object, whose
        ops_to_run are populated with the ops to run in each iteration (by the
        training loop or by objects watching for iteration events), fed into
        into `session.run()` and then passed to the app (and observers) for
        interpretation.
        """

        # Add observers for tensorboard, and console output (move to io?)
        self.tensorboard = TensorBoardLogger(self.outputs_collector,
                                             self.summary_dir, sess.graph,
                                             self.tensorboard_every_n)
        self.console = ConsoleLogger(self.outputs_collector)

        # Core training loop handling
        def add_gradient(iter_msg):
            """ Event handler to add the backpropagation update.
            iter_msg is an IterationMessage object """
            iter_msg.ops_to_run['gradients'] = self.app.gradient_op
        self.pre_train_iter.connect(add_gradient)
        self._loop(self.interleaved_iteration_generator(), sess, loop_status)

    def _inference_loop(self, sess, loop_status):
        """
        Runs all variables returned by outputs_collector,
        this loop stops when the return value of
        application.interpret_output is False.
        """

        loop_status['all_saved_flag'] = False

        self.console = ConsoleLogger(self.outputs_collector)

        def is_complete(iter_msg):
            """ Event handler to trigger the completion message.
            iter_msg is an IterationMessage object """
            if not iter_msg.status:
                tf.logging.info('processed all batches.')
                loop_status['all_saved_flag'] = True
                iter_msg.should_stop = True
        self.post_infer_iter.connect(is_complete)

        self._loop(iter_generator(itertools.count(), INFER), sess, loop_status)

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
                #raise ValueError
            return '/{}:{}'.format(device, device_id)
        # in inference: use gpu for everything whenever n_local_gpus
        return '/gpu:0' if n_local_gpus > 0 else '/cpu:0'

    @staticmethod
    def _create_app(app_type_string):
        """
        Import the application module
        """
        app_class = ApplicationFactory.create(app_type_string)
        app_class.clear()
        return app_class

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
    """ Generate a numbered sequence of IterationMessage objects
    with phase-appropriate signals.
    count_generator is an iterable object yielding iteration numbers
    phase is one of TRAIN, VALID or INFER
    """
    signals = {TRAIN: (ApplicationDriver.pre_train_iter,
                       ApplicationDriver.post_train_iter),
               VALID: (ApplicationDriver.pre_validation_iter,
                       ApplicationDriver.post_validation_iter),
               INFER: (ApplicationDriver.pre_infer_iter,
                       ApplicationDriver.post_infer_iter)}
    for iter_i in count_generator:
        iter_msg = IterationMessage()
        iter_msg.current_iter, iter_msg.phase = iter_i, phase
        iter_msg.pre_iter = signals[phase][0]
        iter_msg.post_iter = signals[phase][1]
        yield iter_msg


class ConsoleLogger(object):
    """ This class handles iteration events to print output to the console """
    def __init__(self, outputs_collector):
        self.outputs_collector = outputs_collector
        ApplicationDriver.pre_train_iter.connect(self.on_pre_iter)
        ApplicationDriver.pre_validation_iter.connect(self.on_pre_iter)
        ApplicationDriver.post_train_iter.connect(self.on_post_iter)
        ApplicationDriver.post_validation_iter.connect(self.on_post_iter)
        ApplicationDriver.post_infer_iter.connect(self.on_post_iter)

    def on_pre_iter(self, iter_msg):
        """ Event handler to add all the console output ops to the iteration
        message.
        iter_msg is an IterationMessage object """
        iter_msg.ops_to_run[CONSOLE] = \
            self.outputs_collector.variables(CONSOLE)

    # pylint: disable=no-self-use
    def on_post_iter(self, iter_msg):
        """ Event handler to log the console outputs from the iteration
        message.
        iter_msg is an IterationMessage object """
        tf.logging.info(iter_msg.to_console_string())


class ModelSaver(object):
    """This class handles iteration events to save the model at regular
    intervals and at the end of training."""
    def __init__(self, sess, saver, save_every_n, session_prefix):
        self.sess = sess
        self.session_prefix = session_prefix
        self.save_every_n = save_every_n
        self.saver = saver
        ApplicationDriver.post_training.connect(self.save_model)
        ApplicationDriver.post_train_iter.connect(self.save_model_interval)

    def save_model_interval(self, iter_msg):
        """ Save the model if the iteration at at the specified interval
        iter_msg is an IterationMessage object """
        if iter_msg.current_iter > 0 and \
                self.save_every_n > 0 and \
                (iter_msg.current_iter % self.save_every_n == 0):
            self.save_model(iter_msg)

    def save_model(self, iter_msg):
        """ Save the model regardless of iteration
        iter_msg is an IterationMessage object """
        self.saver.save(sess=self.sess,
                        save_path=self.session_prefix,
                        global_step=iter_msg.current_iter)
        tf.logging.info('iter %d saved: %s', iter_msg.current_iter,
                        self.session_prefix)

class TensorBoardLogger(object):
    """ This class handles iteration events to log summaries to the
    TensorBoard log."""
    def __init__(self, outputs_collector, summary_dir, graph,
                 tensorboard_every_n):
        self.summary_dir = summary_dir
        self.writer_train = tf.summary.FileWriter(
            os.path.join(self.summary_dir, TRAIN), graph)
        self.writer_valid = tf.summary.FileWriter(
            os.path.join(self.summary_dir, VALID), graph)
        self.outputs_collector = outputs_collector
        self.tensorboard_every_n = tensorboard_every_n
        ApplicationDriver.pre_train_iter.connect(self.on_pre_iter)
        ApplicationDriver.pre_validation_iter.connect(self.on_pre_iter)
        ApplicationDriver.post_train_iter.connect(self.on_post_train_iter)
        ApplicationDriver.post_validation_iter.connect(
            self.on_post_validation_iter)

    def filter(self, iter_msg):
        """ Decide whether to save a TensorBoard log entry for a given
        iteration.
        iter_msg is an IterationMessage object """
        return self.tensorboard_every_n > 0 and \
            (iter_msg.current_iter % self.tensorboard_every_n == 0)

    def on_pre_iter(self, iter_msg):
        """ Event handler to add all the TensorBoard summaries to the iteration
        message.
        iter_msg is an IterationMessage object """
        if self.filter(iter_msg):
            iter_msg.ops_to_run[TF_SUMMARIES] = \
                self.outputs_collector.variables(TF_SUMMARIES)

    def on_post_train_iter(self, iter_msg):
        """ Event handler to write the training TensorBoard log entry to disk.
        iter_msg is an IterationMessage object """
        if self.filter(iter_msg):
            iter_msg.to_tf_summary(self.writer_train)

    def on_post_validation_iter(self, iter_msg):
        """ Event handler to write the valiation TensorBoard log entry to disk.
        iter_msg is an IterationMessage object """
        if self.filter(iter_msg):
            iter_msg.to_tf_summary(self.writer_valid)
