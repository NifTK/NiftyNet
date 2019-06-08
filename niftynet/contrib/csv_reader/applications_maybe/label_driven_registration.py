"""
A preliminary re-implementation of:
    Hu et al., Weakly-Supervised Convolutional Neural Networks for
    Multimodal Image Registration, Medical Image Analysis (2018)
    https://doi.org/10.1016/j.media.2018.07.002

The original implementation and tutorial is available at:
    https://github.com/YipengHu/label-reg
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.io.image_reader import ImageReader
from niftynet.contrib.sampler_pairwise.sampler_pairwise_uniform import \
    PairwiseUniformSampler
from niftynet.contrib.sampler_pairwise.sampler_pairwise_resize import \
    PairwiseResizeSampler
from niftynet.contrib.csv_reader.csv_reader import CSVReader
from niftynet.engine.application_factory import \
    OptimiserFactory, ApplicationNetFactory
from niftynet.engine.application_variables import \
    NETWORK_OUTPUT, CONSOLE, TF_SUMMARIES
from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator

from niftynet.layer.resampler import ResamplerLayer
from niftynet.layer.pad import PadLayer
from niftynet.layer.loss_segmentation import LossFunction


SUPPORTED_INPUT = {'moving_image', 'moving_label',
                   'fixed_image', 'fixed_label'}


class RegApp(BaseApplication):

    REQUIRED_CONFIG_SECTION = "REGISTRATION"

    def __init__(self, net_param, action_param, action):
        BaseApplication.__init__(self)
        tf.logging.info('starting label-driven registration')
        self.action = action

        self.net_param = net_param
        self.action_param = action_param

        self.registration_param = None
        self.data_param = None

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        self.data_param = data_param
        self.registration_param = task_param

        if self.is_evaluation:
            NotImplementedError('Evaluation is not yet '
                                'supported in this application.')
        try:
            reader_phase = self.action_param.dataset_to_infer
        except AttributeError:
            reader_phase = None
        file_lists = data_partitioner.get_file_lists_by(
            phase=reader_phase, action=self.action)

        self.readers = []
        for file_list in file_lists:
            fixed_reader = ImageReader({'fixed_image', 'fixed_label'})
            fixed_reader.initialise(data_param, task_param, file_list)
            self.readers.append(fixed_reader)

            moving_reader = ImageReader({'moving_image', 'moving_label'})
            moving_reader.initialise(data_param, task_param, file_list)
            self.readers.append(moving_reader)

        # pad the fixed target only
        # moving image will be resampled to match the targets
        #volume_padding_layer = []
        #if self.net_param.volume_padding_size:
        #    volume_padding_layer.append(PadLayer(
        #        image_name=('fixed_image', 'fixed_label'),
        #        border=self.net_param.volume_padding_size))

        #for reader in self.readers:
        #    reader.add_preprocessing_layers(volume_padding_layer)


    def initialise_sampler(self):
        if self.is_training:
            self.sampler = []
            assert len(self.readers) >= 2, 'at least two readers are required'
            training_sampler = PairwiseUniformSampler(
                reader_0=self.readers[0],
                reader_1=self.readers[1],
                data_param=self.data_param,
                batch_size=self.net_param.batch_size)
            self.sampler.append(training_sampler)
            # adding validation readers if possible
            if len(self.readers) >= 4:
                validation_sampler = PairwiseUniformSampler(
                    reader_0=self.readers[2],
                    reader_1=self.readers[3],
                    data_param=self.data_param,
                    batch_size=self.net_param.batch_size)
                self.sampler.append(validation_sampler)
        else:
            self.sampler = PairwiseResizeSampler(
                reader_0=self.readers[0],
                reader_1=self.readers[1],
                data_param=self.data_param,
                batch_size=self.net_param.batch_size)

    def initialise_network(self):
        decay = self.net_param.decay
        self.net = ApplicationNetFactory.create(self.net_param.name)(decay)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):

        def switch_samplers(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0 if for_training else -1]
                return sampler()  # returns image only

        if self.is_training:
            self.patience = self.action_param.patience
            if self.action_param.validation_every_n > 0:
                sampler_window = \
                    tf.cond(tf.logical_not(self.is_validation),
                            lambda: switch_samplers(True),
                            lambda: switch_samplers(False))
            else:
                sampler_window = switch_samplers(True)

            image_windows, _ = sampler_window
            # image_windows, locations = sampler_window

            # decode channels for moving and fixed images
            image_windows_list = [
                tf.expand_dims(img, axis=-1)
                for img in tf.unstack(image_windows, axis=-1)]
            fixed_image, fixed_label, moving_image, moving_label = \
                image_windows_list

            # estimate ddf
            dense_field = self.net(fixed_image, moving_image)
            if isinstance(dense_field, tuple):
                dense_field = dense_field[0]

            # transform the moving labels
            resampler = ResamplerLayer(
                interpolation='linear', boundary='replicate')
            resampled_moving_label = resampler(moving_label, dense_field)

            # compute label loss (foreground only)
            loss_func = LossFunction(
                n_class=1,
                loss_type=self.action_param.loss_type,
                softmax=False)
            label_loss = loss_func(prediction=resampled_moving_label,
                                   ground_truth=fixed_label)

            dice_fg = 1.0 - label_loss
            # appending regularisation loss
            total_loss = label_loss
            reg_loss = tf.get_collection('bending_energy')
            if reg_loss:
                total_loss = total_loss + \
                    self.net_param.decay * tf.reduce_mean(reg_loss)

            self.total_loss = total_loss

            # compute training gradients
            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)
            grads = self.optimiser.compute_gradients(
                total_loss, colocate_gradients_with_ops=True)
            gradients_collector.add_to_collection(grads)

            metrics_dice = loss_func(
                prediction=tf.to_float(resampled_moving_label >= 0.5),
                ground_truth=tf.to_float(fixed_label >= 0.5))
            metrics_dice = 1.0 - metrics_dice

            # command line output
            outputs_collector.add_to_collection(
                var=dice_fg, name='one_minus_data_loss',
                collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=tf.reduce_mean(reg_loss), name='bending_energy',
                collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=total_loss, name='total_loss', collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=metrics_dice, name='ave_fg_dice', collection=CONSOLE)

            # for tensorboard
            outputs_collector.add_to_collection(
                var=dice_fg,
                name='data_loss',
                average_over_devices=True,
                summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=total_loss,
                name='total_loss',
                average_over_devices=True,
                summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=metrics_dice,
                name='averaged_foreground_Dice',
                average_over_devices=True,
                summary_type='scalar',
                collection=TF_SUMMARIES)

            # for visualisation debugging
            # resampled_moving_image = resampler(moving_image, dense_field)
            # outputs_collector.add_to_collection(
            #     var=fixed_image, name='fixed_image',
            #     collection=NETWORK_OUTPUT)
            # outputs_collector.add_to_collection(
            #     var=fixed_label, name='fixed_label',
            #     collection=NETWORK_OUTPUT)
            # outputs_collector.add_to_collection(
            #     var=moving_image, name='moving_image',
            #     collection=NETWORK_OUTPUT)
            # outputs_collector.add_to_collection(
            #     var=moving_label, name='moving_label',
            #     collection=NETWORK_OUTPUT)
            # outputs_collector.add_to_collection(
            #     var=resampled_moving_image, name='resampled_image',
            #     collection=NETWORK_OUTPUT)
            # outputs_collector.add_to_collection(
            #     var=resampled_moving_label, name='resampled_label',
            #     collection=NETWORK_OUTPUT)
            # outputs_collector.add_to_collection(
            #     var=dense_field, name='ddf', collection=NETWORK_OUTPUT)
            # outputs_collector.add_to_collection(
            #     var=locations, name='locations', collection=NETWORK_OUTPUT)

            # outputs_collector.add_to_collection(
            #     var=shift[0], name='a', collection=CONSOLE)
            # outputs_collector.add_to_collection(
            #     var=shift[1], name='b', collection=CONSOLE)
        else:
            image_windows, locations = self.sampler()
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
            resampled_moving_image = resampler(moving_image, dense_field)
            resampled_moving_label = resampler(moving_label, dense_field)

            outputs_collector.add_to_collection(
                var=fixed_image, name='fixed_image',
                collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=moving_image, name='moving_image',
                collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=resampled_moving_image,
                name='resampled_moving_image',
                collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=resampled_moving_label,
                name='resampled_moving_label',
                collection=NETWORK_OUTPUT)

            outputs_collector.add_to_collection(
                var=fixed_label, name='fixed_label',
                collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=moving_label, name='moving_label',
                collection=NETWORK_OUTPUT)
            #outputs_collector.add_to_collection(
            #    var=dense_field, name='field',
            #    collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=locations, name='locations',
                collection=NETWORK_OUTPUT)

            self.output_decoder = ResizeSamplesAggregator(
                image_reader=self.readers[0], # fixed image reader
                name='fixed_image',
                output_path=self.action_param.save_seg_dir,
                interp_order=self.action_param.output_interp_order)

    def interpret_output(self, batch_output):
        if self.is_training:
            return True
        return self.output_decoder.decode_batch(
            {'window_resampled':batch_output['resampled_moving_image']},
            batch_output['locations'])

