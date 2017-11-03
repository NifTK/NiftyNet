from __future__ import absolute_import, division, print_function

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.sampler_uniform import UniformSampler
from niftynet.io.image_reader import ImageReader
from niftynet.contrib.sampler_pairwise.sampler_pairwise import PairwiseSampler
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_factory import ApplicationNetFactory
from niftynet.engine.application_variables import NETWORK_OUTPUT


SUPPORTED_INPUT = {'moving_image', 'moving_label',
                   'fixed_image', 'fixed_label'}


class RegApp(BaseApplication):

    REQUIRED_CONFIG_SECTION = "REGISTRATION"

    def __init__(self, net_param, action_param, is_training):
        BaseApplication.__init__(self)
        tf.logging.info('starting label-driven registration')
        self.is_training = is_training

        self.net_param = net_param
        self.action_param = action_param

        self.registration_param = None
        self.data_param = None

    def initialise_dataset_loader(self, data_param=None, task_param=None):
        self.data_param = data_param
        self.registration_param = task_param

        if self.is_training:
            self.reader = (ImageReader({'fixed_image', 'fixed_label'}),
                           ImageReader({'moving_image', 'moving_label'}))
        else:
            raise NotImplementedError
        for reader in self.reader:
            reader.initialise_reader(data_param, task_param)

    def initialise_sampler(self):
        if self.is_training:
            self.sampler = [
                PairwiseSampler(
                    reader_0=self.reader[0],
                    reader_1=self.reader[1],
                    data_param=self.data_param,
                    batch_size=self.net_param.batch_size,
                    window_per_image=self.action_param.sample_per_volume)]
        else:
            raise NotImplementedError

    def initialise_network(self):
        decay = self.net_param.decay
        self.net = ApplicationNetFactory.create(self.net_param.name)(decay)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        if self.is_training:
            image_windows = self.sampler[0]()
            image_windows_list = [
                tf.expand_dims(img, axis=-1)
                for img in tf.unstack(image_windows, axis=-1)]
            fixed_image, fixed_label, moving_image, moving_label = \
                image_windows_list
            net_out = self.net(fixed_image, moving_image)

            outputs_collector.add_to_collection(
                var=net_out,
                name='test_sampler_output',
                collection=NETWORK_OUTPUT)

            self.dummy_op = tf.constant(1.0)
            a = tf.get_variable('a', shape=[1])
            loss = tf.constant(1.0) + a
            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)
            grads = self.optimiser.compute_gradients(loss)
            gradients_collector.add_to_collection(grads)
        else:
            raise NotImplementedError

    def interpret_output(self, batch_output):
        if self.is_training:
            print(batch_output['test_sampler_output'].shape)
            #windows = batch_output['test_sampler_output']
            #import matplotlib.pyplot as plt
            #plt.imshow(windows[1, 20, ..., 0])
            #plt.show()
            return True
        else:
            raise NotImplementedError

    def training_ops(self, start_iter=0, end_iter=1):
        end_iter = max(start_iter, end_iter)
        for iter_i in range(start_iter, end_iter):
            yield iter_i, self.dummy_op
