from __future__ import absolute_import, division, print_function

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.sampler_uniform import UniformSampler
from niftynet.io.image_reader import ImageReader
from niftynet.contrib.sampler_pairwise.sampler_pairwise import PairwiseSampler
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_factory import ApplicationNetFactory
from niftynet.engine.application_variables import NETWORK_OUTPUT
from niftynet.layer.resampler import ResamplerLayer
#from niftynet.layer.loss_segmentation import LossFunction
from niftynet.engine.application_variables import CONSOLE
from niftynet.layer.loss_regression import LossFunction


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

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        self.data_param = data_param
        self.registration_param = task_param

        # add validation here
        if self.is_training:
            file_lists = [data_partitioner.train_files]
            self.readers = (ImageReader({'fixed_image', 'fixed_label'}),
                            ImageReader({'moving_image', 'moving_label'}))
        else:
            raise NotImplementedError

        for file_list in file_lists:
            for reader in self.readers:
                reader.initialise(data_param, task_param, file_list)

    def initialise_sampler(self):
        if self.is_training:
            self.sampler = [[
                PairwiseSampler(
                    reader_0=self.readers[0],
                    reader_1=self.readers[1],
                    data_param=self.data_param,
                    batch_size=self.net_param.batch_size,
                    window_per_image=self.action_param.sample_per_volume)]]
        else:
            raise NotImplementedError

    def initialise_network(self):
        decay = self.net_param.decay
        self.net = ApplicationNetFactory.create(self.net_param.name)(decay)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        if self.is_training:
            image_windows = self.sampler[0][0]()
            image_windows_list = [
                tf.expand_dims(img, axis=-1)
                for img in tf.unstack(image_windows, axis=-1)]
            fixed_image, fixed_label, moving_image, moving_label = \
                image_windows_list
            dense_field = self.net(fixed_image, moving_image)
            if isinstance(dense_field, tuple):
                dense_field = dense_field[0]
            # transform the moving labels
            resampler = ResamplerLayer(
                interpolation='linear', boundary='replicate')
            resampled_moving_label = resampler(moving_label, dense_field)
            # compute label loss
            loss_func = LossFunction(loss_type='L2Loss')
            label_loss = loss_func(
                prediction=resampled_moving_label,
                ground_truth=fixed_label)
            outputs_collector.add_to_collection(
                var=label_loss,
                name='label_loss',
                collection=CONSOLE)

            # compute training gradients
            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)
            grads = self.optimiser.compute_gradients(label_loss)
            gradients_collector.add_to_collection(grads)
        else:
            raise NotImplementedError

    def interpret_output(self, batch_output):
        if self.is_training:
            return True
        raise NotImplementedError

