# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import ApplicationNetFactory
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.engine.sampler_random_vector_v2 import RandomVectorSampler
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.contrib.csv_reader.sampler_resize_v2_csv import \
    ResizeSamplerCSV as ResizeSampler
from niftynet.engine.windows_aggregator_identity import WindowAsImageAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.loss_gan import LossFunction
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer

SUPPORTED_INPUT = set(['image', 'conditioning'])


class GANApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = "GAN"

    def __init__(self, net_param, action_param, action):
        BaseApplication.__init__(self)
        tf.logging.info('starting GAN application')
        self.action = action

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.gan_param = None

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        self.data_param = data_param
        self.gan_param = task_param

        if self.is_training:
            reader_names = ('image', 'conditioning')
        elif self.is_inference:
            # in the inference process use `conditioning` input only
            reader_names = ('conditioning',)
        elif self.is_evaluation:
            tf.logging.fatal(
                'Evaluation is not yet supported in this application.')
            raise NotImplementedError
        else:
            tf.logging.fatal(
                'Action `%s` not supported. Expected one of %s',
                self.action, self.SUPPORTED_PHASES)
            raise ValueError
        try:
            reader_phase = self.action_param.dataset_to_infer
        except AttributeError:
            reader_phase = None
        file_lists = data_partitioner.get_file_lists_by(
            phase=reader_phase, action=self.action)
        self.readers = [
            ImageReader(reader_names).initialise(
                data_param, task_param, file_list) for file_list in file_lists]

        # initialise input preprocessing layers
        foreground_masking_layer = BinaryMaskingLayer(
            type_str=self.net_param.foreground_type,
            multimod_fusion=self.net_param.multimod_foreground_type,
            threshold=0.0) \
            if self.net_param.normalise_foreground_only else None
        mean_var_normaliser = MeanVarNormalisationLayer(
            image_name='image', binary_masking_func=foreground_masking_layer) \
            if self.net_param.whitening else None
        histogram_normaliser = HistogramNormalisationLayer(
            image_name='image',
            modalities=vars(task_param).get('image'),
            model_filename=self.net_param.histogram_ref_file,
            binary_masking_func=foreground_masking_layer,
            norm_type=self.net_param.norm_type,
            cutoff=self.net_param.cutoff,
            name='hist_norm_layer') \
            if (self.net_param.histogram_ref_file and
                self.net_param.normalisation) else None

        normalisation_layers = []
        if histogram_normaliser is not None:
            normalisation_layers.append(histogram_normaliser)
        if mean_var_normaliser is not None:
            normalisation_layers.append(mean_var_normaliser)

        # initialise training data augmentation layers
        augmentation_layers = []
        if self.is_training:
            if self.action_param.random_flipping_axes != -1:
                augmentation_layers.append(RandomFlipLayer(
                    flip_axes=self.action_param.random_flipping_axes))
            if self.action_param.scaling_percentage:
                augmentation_layers.append(RandomSpatialScalingLayer(
                    min_percentage=self.action_param.scaling_percentage[0],
                    max_percentage=self.action_param.scaling_percentage[1],
                    antialiasing=self.action_param.antialiasing,
                    isotropic=self.action_param.isotropic_scaling))
            if self.action_param.rotation_angle:
                augmentation_layers.append(RandomRotationLayer())
                augmentation_layers[-1].init_uniform_angle(
                    self.action_param.rotation_angle)

        # only add augmentation to first reader (not validation reader)
        self.readers[0].add_preprocessing_layers(
            normalisation_layers + augmentation_layers)

        for reader in self.readers[1:]:
            reader.add_preprocessing_layers(normalisation_layers)

    def initialise_sampler(self):
        self.sampler = []
        if self.is_training:
            self.sampler.append([ResizeSampler(
                reader=reader,
                window_sizes=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=1,
                shuffle=True,
                queue_length=self.net_param.queue_length) for reader in
                self.readers])
        else:
            self.sampler.append([RandomVectorSampler(
                names=('vector',),
                vector_size=(self.gan_param.noise_size,),
                batch_size=self.net_param.batch_size,
                n_interpolations=self.gan_param.n_interpolations,
                repeat=None,
                queue_length=self.net_param.queue_length) for _ in
                self.readers])
            # repeat each resized image n times, so that each
            # image matches one random vector,
            # (n = self.gan_param.n_interpolations)
            self.sampler.append([ResizeSampler(
                reader=reader,
                window_sizes=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=self.gan_param.n_interpolations,
                shuffle=False,
                queue_length=self.net_param.queue_length) for reader in
                self.readers])

    def initialise_network(self):
        self.net = ApplicationNetFactory.create(self.net_param.name)()

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        if self.is_training:
            self.patience = self.action_param.patience

            def switch_sampler(for_training):
                with tf.name_scope('train' if for_training else 'validation'):
                    sampler = self.get_sampler()[0][0 if for_training else -1]
                    return sampler.pop_batch_op()

            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(for_training=True),
                                    lambda: switch_sampler(for_training=False))
            else:
                data_dict = switch_sampler(for_training=True)

            images = tf.cast(data_dict['image'], tf.float32)
            noise_shape = [self.net_param.batch_size,
                           self.gan_param.noise_size]
            noise = tf.random_normal(shape=noise_shape,
                                     mean=0.0,
                                     stddev=1.0,
                                     dtype=tf.float32)
            conditioning = data_dict['conditioning']
            net_output = self.net(
                noise, images, conditioning, self.is_training)

            loss_func = LossFunction(
                loss_type=self.action_param.loss_type)
            real_logits = net_output[1]
            fake_logits = net_output[2]
            lossG, lossD = loss_func(real_logits, fake_logits)
            if self.net_param.decay > 0:
                reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                if reg_losses:
                    reg_loss = tf.reduce_mean(
                        [tf.reduce_mean(l_reg) for l_reg in reg_losses])
                    lossD = lossD + reg_loss
                    lossG = lossG + reg_loss

            self.total_loss = lossD + lossG

            outputs_collector.add_to_collection(
                var=self.total_loss, name='total_loss',
                average_over_devices=True, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.total_loss, name='total_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            # variables to display in STDOUT
            outputs_collector.add_to_collection(
                var=lossD, name='lossD', average_over_devices=True,
                collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=lossG, name='lossG', average_over_devices=False,
                collection=CONSOLE)
            # variables to display in tensorboard
            outputs_collector.add_to_collection(
                var=lossG, name='lossG', average_over_devices=False,
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=lossG, name='lossD', average_over_devices=True,
                collection=TF_SUMMARIES)

            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)

            with tf.name_scope('ComputeGradients'):
                # gradients of generator
                generator_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                generator_grads = self.optimiser.compute_gradients(
                    lossG,
                    var_list=generator_variables,
                    colocate_gradients_with_ops=True)

                # gradients of discriminator
                discriminator_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                discriminator_grads = self.optimiser.compute_gradients(
                    lossD,
                    var_list=discriminator_variables,
                    colocate_gradients_with_ops=True)
                grads = [generator_grads, discriminator_grads]

                # add the grads back to application_driver's training_grads
                gradients_collector.add_to_collection(grads)
        else:
            data_dict = self.get_sampler()[0][0].pop_batch_op()
            conditioning_dict = self.get_sampler()[1][0].pop_batch_op()
            conditioning = conditioning_dict['conditioning']
            image_size = conditioning.shape.as_list()[:-1]
            dummy_image = tf.zeros(image_size + [1])
            net_output = self.net(data_dict['vector'],
                                  dummy_image,
                                  conditioning,
                                  self.is_training)
            outputs_collector.add_to_collection(
                var=net_output[0],
                name='image',
                average_over_devices=False,
                collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=conditioning_dict['conditioning_location'],
                name='location',
                average_over_devices=False,
                collection=NETWORK_OUTPUT)

            self.output_decoder = WindowAsImageAggregator(
                image_reader=self.readers[0],
                output_path=self.action_param.save_seg_dir)

    def interpret_output(self, batch_output):
        if self.is_training:
            return True
        return self.output_decoder.decode_batch(
            {'window_image': batch_output['image']},
            batch_output['location'])
