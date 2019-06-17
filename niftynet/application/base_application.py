# -*- coding: utf-8 -*-
"""
Interface of NiftyNet application
"""
import os
from argparse import Namespace

import tensorflow as tf
from six import with_metaclass

from niftynet.engine.signal import TRAIN, INFER, EVAL

APP_INSTANCE = None  # global so it can be reset


# pylint: disable=global-statement
class SingletonApplication(type):
    """
    Maintaining a global application instance.
    """
    def __call__(cls, *args, **kwargs):
        global APP_INSTANCE
        if APP_INSTANCE is None:
            APP_INSTANCE = \
                super(SingletonApplication, cls).__call__(*args, **kwargs)
        # else:
        #     raise RuntimeError('application instance already started.')
        return APP_INSTANCE

    @classmethod
    def clear(mcs):
        """
        Remove the instance.
        :return:
        """
        global APP_INSTANCE
        APP_INSTANCE = None


class BaseApplication(with_metaclass(SingletonApplication, object)):
    """
    BaseApplication represents an interface.

    Each application ``type_str`` should support to use
    the standard training and inference driver.
    """

    # defines name of the customised configuration file section
    # the section collects all application specific user parameters
    REQUIRED_CONFIG_SECTION = None

    SUPPORTED_PHASES = {TRAIN, INFER, EVAL}
    _action = TRAIN
    # TF placeholders for switching network on the fly
    is_validation = None

    # input of the network
    readers = None
    sampler = None

    # the network
    net = None

    # training the network
    optimiser = None
    gradient_op = None

    # interpret network output
    output_decoder = None
    outputs_collector = None
    gradients_collector = None

    # performance
    total_loss = None
    patience = None
    performance_history = []
    mode = None

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        """
        this function initialise self.readers

        :param data_param: input modality specifications
        :param task_param: contains task keywords for grouping data_param
        :param data_partitioner:
                           specifies train/valid/infer splitting if needed
        :return:
        """
        raise NotImplementedError

    def initialise_sampler(self):
        """
        Samplers take ``self.reader`` as input and generates
        sequences of ImageWindow that will be fed to the networks

        This function sets ``self.sampler``.
        """
        raise NotImplementedError

    def initialise_network(self):
        """
        This function create an instance of network and sets ``self.net``

        :return: None
        """
        raise NotImplementedError

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        """
        Adding sampler output tensor and network tensors to the graph.

        :param outputs_collector:
        :param gradients_collector:
        :return:
        """
        raise NotImplementedError

    def interpret_output(self, batch_output):
        """
        Implement output interpretations, e.g., save to hard drive
        cache output windows.

        :param batch_output: outputs by running the tf graph
        :return: True indicates the driver should continue the loop
            False indicates the drive should stop
        """
        raise NotImplementedError

    def add_inferred_output_like(self, data_param, task_param, name):
        """ This function adds entries to parameter objects to enable
        the evaluation action to automatically read in the output of a
        previous inference run if inference is not explicitly specified.

        This can be used in an application if there is a data section
        entry in the configuration file that matches the inference output.
        In supervised learning, the reference data section would often
        match the inference output and could be used here. Otherwise,
        a template data section could be used.

        :param data_param:
        :param task_param:
        :param name:  name of input parameter to copy parameters from
        :return: modified data_param and task_param
        """
        print(task_param)
        # Add the data parameter
        if 'inferred' not in data_param:
            data_name = vars(task_param)[name][0]
            inferred_param = Namespace(**vars(data_param[data_name]))
            inferred_param.csv_file = os.path.join(
                self.action_param.save_seg_dir, 'inferred.csv')
            data_param['inferred'] = inferred_param
        # Add the task parameter
        if 'inferred' not in task_param or not task_param.inferred:
            task_param.inferred = ('inferred',)
        return data_param, task_param

    def set_iteration_update(self, iteration_message):
        """
        At each iteration ``application_driver`` calls::

            output = tf.session.run(variables_to_eval, feed_dict=data_dict)

        to evaluate TF graph elements, where
        ``variables_to_eval`` and ``data_dict`` are retrieved from
        ``iteration_message.ops_to_run`` and
        ``iteration_message.data_feed_dict``
         (In addition to the variables collected by self.output_collector).

        The output of `tf.session.run(...)` will be stored at
        ``iteration_message.current_iter_output``, and can be accessed
        from ``engine.handler_network_output.OutputInterpreter``.

        override this function for more complex operations
        (such as learning rate decay) according to
        ``iteration_message.current_iter``.
        """
        if iteration_message.is_training:
            iteration_message.data_feed_dict[self.is_validation] = False
        elif iteration_message.is_validation:
            iteration_message.data_feed_dict[self.is_validation] = True

    def get_sampler(self):
        """
        Get samplers of the application

        :return: ``niftynet.engine.sampler_*`` instances
        """
        return self.sampler

    def add_validation_flag(self):
        """
        Add a TF placeholder for switching between train/valid graphs,
        this function sets ``self.is_validation``. ``self.is_validation``
        can be used by applications.

        :return:
        """
        self.is_validation = \
            tf.placeholder_with_default(False, [], 'is_validation')

    @property
    def action(self):
        """
        A string indicating the action in train/inference/evaluation

        :return:
        """
        return self._action

    @action.setter
    def action(self, value):
        """
        The action should have at least two characters matching
        the one of the phase string TRAIN, INFER, EVAL

        :param value:
        :return:
        """
        try:
            self._action = value.lower()
            assert len(self._action) >= 2
        except (AttributeError, AssertionError):
            tf.logging.fatal('Error setting application action: %s', value)

    @property
    def is_training(self):
        """

        :return: boolean value indicating if the phase is training
        """
        return TRAIN.startswith(self.action)

    @property
    def is_inference(self):
        """

        :return: boolean value indicating if the phase is inference
        """
        return INFER.startswith(self.action)

    @property
    def is_evaluation(self):
        """

        :return: boolean value indicating if the action is evaluation
        """
        return EVAL.startswith(self.action)

    @staticmethod
    def stop():
        """
        remove application instance if there's any.

        :return:
        """
        SingletonApplication.clear()
