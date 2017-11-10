# -*- coding: utf-8 -*-
"""
Interface of NiftyNet application
"""

import tensorflow as tf
from six import with_metaclass

from niftynet.layer.base_layer import TrainableLayer
from niftynet.utilities import util_common
from niftynet.io.image_sets_partitioner import TRAIN, VALID


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
    Each application type_str should support to use
    the standard training and inference driver
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
            raise NotImplementedError('sampler should be initialised')
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
        set samplers take self.reader as input and generates
        sequences of ImageWindow that will be fed to the networks
        This function sets self.sampler
        """
        raise NotImplementedError

    def initialise_network(self):
        """
        This function create an instance of network
        sets self.net
        :return: None
        """
        raise NotImplementedError

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        """
        adding sampler output tensor and network tensors to the graph.
        :param outputs_collector:
        :param gradients_collector:
        :return:
        """
        raise NotImplementedError

    def interpret_output(self, batch_output):
        """
        implement output interpretations, e.g., save to hard drive
        cache output windows
        :param batch_output: outputs by running the tf graph
        :return: True indicates the driver should continue the loop
                 False indicates the drive should stop
        """
        raise NotImplementedError

    def set_network_update_op(self, gradients):
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
        for sampler_set in self.get_sampler():
            for sampler in sampler_set:
                if sampler:
                    sampler.close_all()

    def update(self, iteration_message):
        if iteration_message.phase == TRAIN:
            iteration_message.data_feed_dict[self.is_validation] = False
            iteration_message.ops_to_run = {'grad': self.gradient_op}
        if iteration_message.phase == VALID:
            iteration_message.data_feed_dict[self.is_validation] = True
            iteration_message.ops_to_run = {}


    def get_sampler(self):
        return self.sampler

    def add_validation_flag(self):
        """
        add a TF placeholder for switching between train/valid graphs
        :return:
        """
        self.is_validation = \
            tf.placeholder_with_default(False, [], 'is_validation')
