# -*- coding: utf-8 -*-
"""
Interface of NiftyNet application
"""

import tensorflow as tf
from six import with_metaclass

from niftynet.layer.base_layer import TrainableLayer
from niftynet.utilities import util_common


class SingletonApplication(type):
    _instances = None

    def __call__(cls, *args, **kwargs):
        if cls._instances is None:
            cls._instances = \
                super(SingletonApplication, cls).__call__(*args, **kwargs)
        # else:
        #     raise RuntimeError('application instance already started.')
        return cls._instances


class BaseApplication(with_metaclass(SingletonApplication, object)):
    """
    BaseApplication represents an interface.

    Each application ``type_str`` should support to use
    the standard training and inference driver.
    """

    # defines name of the customised configuration file section
    # the section collects all application specific user parameters
    REQUIRED_CONFIG_SECTION = None

    # boolean flag
    is_training = True
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

    def check_initialisations(self):
        if self.readers is None:
            raise NotImplementedError('reader should be initialised')
        if self.sampler is None:
            raise NotImplementedError(
                'Sampler should be initialised; to disable the sampler, '
                'set self.sampler to [None].')
        if self.net is None:
            raise NotImplementedError('net should be initialised')
        if not isinstance(self.net, TrainableLayer):
            raise ValueError('self.net should be an instance'
                             ' of niftynet.layer.TrainableLayer')
        if self.optimiser is None and self.is_training:
            raise NotImplementedError('optimiser should be initialised')
        if self.gradient_op is None and self.is_training:
            raise NotImplementedError('gradient_op should be initialised')
        if self.output_decoder is None and not self.is_training:
            raise NotImplementedError('output decoder should be initialised')

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

        This function sets self.sampler.
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

    def set_network_gradient_op(self, gradients):
        """
        create gradient op by optimiser.apply_gradients
        this function sets ``self.gradient_op``.

        Override this function for more complex optimisations such as
        using different optimisers for sub-networks.

        :param gradients: processed gradients from the gradient_collector
        :return:
        """
        grad_list_depth = util_common.list_depth_count(gradients)
        if grad_list_depth == 3:
            # nested depth 3 means: gradients list is nested in terms of:
            # list of networks -> list of network variables
            self.gradient_op = [self.optimiser.apply_gradients(grad)
                                for grad in gradients]
        elif grad_list_depth == 2:
            # nested depth 2 means:
            # gradients list is a list of variables
            self.gradient_op = self.optimiser.apply_gradients(gradients)
        else:
            raise NotImplementedError(
                'This app supports updating a network, or a list of networks.')

    def stop(self):
        """
        stop the sampling threads if there's any.

        :return:
        """
        for sampler in util_common.traverse_nested(self.get_sampler()):
            if sampler is None:
                continue
            sampler.close_all()

    def set_iteration_update(self, iteration_message):
        """
        At each iteration ``application_driver`` calls:
            ``output = tf.session.run(variables_to_eval, feed_dict=data_dict)``

        to evaluate TF graph elements, where
        ``variables_to_eval`` and ``data_dict`` are retrieved from
        ``application_iteration.IterationMessage.ops_to_run`` and
        ``application_iteration.IterationMessage.data_feed_dict``.
        in addition to the variables collected by output_collector;
        implemented in ``application_driver.run_vars``)

        This function (is called before ``tf.session.run`` by the
        driver) provides an interface for accessing ``variables_to_eval`` and
        ``data_dict`` at each iteration.

        Override this function for more complex operations according to
        ``application_iteration.IterationMessage.current_iter``.
        """
        if iteration_message.is_training:
            iteration_message.data_feed_dict[self.is_validation] = False
        elif iteration_message.is_validation:
            iteration_message.data_feed_dict[self.is_validation] = True

    def get_sampler(self):
        """
        get samplers of the application

        :return: ``niftynet.engine.sampler_*`` instances
        """
        return self.sampler

    def add_validation_flag(self):
        """
        add a TF placeholder for switching between train/valid graphs,
        this function sets ``self.is_validation``. ``self.is_validation``
        can be used by applications.

        :return:
        """
        self.is_validation = \
            tf.placeholder_with_default(False, [], 'is_validation')
