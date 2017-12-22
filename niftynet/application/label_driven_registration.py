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
from niftynet.layer.loss_segmentation import LossFunction
from niftynet.layer.pad import PadLayer
from niftynet.engine.application_variables import CONSOLE
#from niftynet.layer.loss_regression import LossFunction


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

        # pad the fixed target only
        volume_padding_layer = []
        if self.net_param.volume_padding_size:
            volume_padding_layer.append(PadLayer(
                image_name=('fixed_image', 'fixed_label'),
                border=self.net_param.volume_padding_size))
        self.readers[0].add_preprocessing_layers(volume_padding_layer)


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
            with tf.device('/cpu:0'):
                image_windows, shift = self.sampler[0][0]()
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
            resampled_moving_label = tf.concat(
                [-resampled_moving_label, resampled_moving_label], axis=-1)
            #resampled_moving_image = resampler(moving_image, dense_field)
            #resampled_moving_label = moving_label * tf.get_variable('a', [1])
            # compute label loss
            loss_func = LossFunction(n_class=2, loss_type='Dice')
            label_loss = loss_func(
                prediction=resampled_moving_label,
                ground_truth=fixed_label)

            reg_loss = tf.get_collection('bending_energy')
            if reg_loss:
                lambda_bending = 0.1
                total_loss = label_loss + lambda_bending * tf.reduce_mean(reg_loss)
                outputs_collector.add_to_collection(
                    var=total_loss, name='total_loss', collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=label_loss, name='label_loss', collection=CONSOLE)

            # compute training gradients
            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)
            grads = self.optimiser.compute_gradients(label_loss)
            gradients_collector.add_to_collection(grads)

            # for visualisation debugging
            #outputs_collector.add_to_collection(
            #    var=fixed_image, name='image', collection=NETWORK_OUTPUT)
            #outputs_collector.add_to_collection(
            #    var=fixed_label, name='label', collection=NETWORK_OUTPUT)
            #outputs_collector.add_to_collection(
            #    var=moving_image, name='moving_image', collection=NETWORK_OUTPUT)
            #outputs_collector.add_to_collection(
            #    var=moving_label, name='moving_label', collection=NETWORK_OUTPUT)
            #outputs_collector.add_to_collection(
            #    var=resampled_moving_label, name='resampled', collection=NETWORK_OUTPUT)
            #outputs_collector.add_to_collection(
            #    var=dense_field, name='ddf', collection=NETWORK_OUTPUT)

            #outputs_collector.add_to_collection(
            #    var=shift[0], name='a', collection=CONSOLE)
            #outputs_collector.add_to_collection(
            #    var=shift[1], name='b', collection=CONSOLE)
        else:
            raise NotImplementedError

    def interpret_output(self, batch_output):
        #import matplotlib.pyplot as plt
        #import pdb; pdb.set_trace()
        if self.is_training:
            return True
        raise NotImplementedError

