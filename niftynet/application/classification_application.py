# -*- coding: utf-8 -*-
"""
This module defines an image-level classification application
that maps from images to scalar, multi-class labels.

This class is instantiated and initalized by the application_driver.
"""

import os

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import \
    ApplicationNetFactory, InitializerFactory, OptimiserFactory
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.layer.loss_classification import LossFunction
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.evaluation.classification_evaluator import ClassificationEvaluator

SUPPORTED_INPUT = set(['image', 'label', 'sampler', 'inferred'])


class ClassificationApplication(BaseApplication):
    """This class defines an application for image-level classification
    problems mapping from images to scalar labels.

    This is the application class to be instantiated by the driver
    and referred to in configuration files.

    Although structurally similar to segmentation, this application
    supports different samplers/aggregators (because patch-based
    processing is not appropriate), and monitoring metrics."""

    REQUIRED_CONFIG_SECTION = "CLASSIFICATION"

    def __init__(self, net_param, action_param, action):
        super(ClassificationApplication, self).__init__()
        tf.logging.info('starting classification application')
        self.action = action

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.classification_param = None
        self.SUPPORTED_SAMPLING = {
            'resize': (self.initialise_resize_sampler,
                       self.initialise_resize_sampler),
        }

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):

        self.data_param = data_param
        self.classification_param = task_param

        if self.is_training:
            reader_names = ('image', 'label', 'sampler')
        elif self.is_inference:
            reader_names = ('image',)
        elif self.is_evaluation:
            reader_names = ('image', 'label', 'inferred')
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

        label_normaliser = DiscreteLabelNormalisationLayer(
            image_name='label',
            modalities=vars(task_param).get('label'),
            model_filename=self.net_param.histogram_ref_file) \
            if (self.net_param.histogram_ref_file and
                task_param.label_normalisation) else None

        normalisation_layers = []
        if histogram_normaliser is not None:
            normalisation_layers.append(histogram_normaliser)
        if mean_var_normaliser is not None:
            normalisation_layers.append(mean_var_normaliser)
        if label_normaliser is not None:
            normalisation_layers.append(label_normaliser)

        augmentation_layers = []
        if self.is_training:
            train_param = self.action_param
            if train_param.random_flipping_axes != -1:
                augmentation_layers.append(RandomFlipLayer(
                    flip_axes=train_param.random_flipping_axes))
            if train_param.scaling_percentage:
                augmentation_layers.append(RandomSpatialScalingLayer(
                    min_percentage=train_param.scaling_percentage[0],
                    max_percentage=train_param.scaling_percentage[1],
                    antialiasing=train_param.antialiasing,
                    isotropic=train_param.isotropic_scaling))
            if train_param.rotation_angle or \
                    self.action_param.rotation_angle_x or \
                    self.action_param.rotation_angle_y or \
                    self.action_param.rotation_angle_z:
                rotation_layer = RandomRotationLayer()
                if train_param.rotation_angle:
                    rotation_layer.init_uniform_angle(
                        train_param.rotation_angle)
                else:
                    rotation_layer.init_non_uniform_angle(
                        self.action_param.rotation_angle_x,
                        self.action_param.rotation_angle_y,
                        self.action_param.rotation_angle_z)
                augmentation_layers.append(rotation_layer)

        # only add augmentation to first reader (not validation reader)
        self.readers[0].add_preprocessing_layers(
             normalisation_layers + augmentation_layers)

        for reader in self.readers[1:]:
            reader.add_preprocessing_layers(normalisation_layers)

        # Checking num_classes is set correctly
        if self.classification_param.num_classes <= 1:
            raise ValueError("Number of classes must be at least 2 for classification")
        for preprocessor in self.readers[0].preprocessors:
            if preprocessor.name == 'label_norm':
                if len(preprocessor.label_map[preprocessor.key[0]]) != self.classification_param.num_classes:
                    raise ValueError("Number of unique labels must be equal to "
                                     "number of classes (check histogram_ref file)")

    def initialise_resize_sampler(self):
        self.sampler = [[ResizeSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            shuffle=self.is_training,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_aggregator(self):
        self.output_decoder = ResizeSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            postfix=self.action_param.output_postfix)

    def initialise_sampler(self):
        if self.is_training:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][0]()
        else:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][1]()

    def initialise_network(self):
        w_regularizer = None
        b_regularizer = None
        reg_type = self.net_param.reg_type.lower()
        decay = self.net_param.decay
        if reg_type == 'l2' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l2_regularizer(decay)
            b_regularizer = regularizers.l2_regularizer(decay)
        elif reg_type == 'l1' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l1_regularizer(decay)
            b_regularizer = regularizers.l1_regularizer(decay)

        self.net = ApplicationNetFactory.create(self.net_param.name)(
            num_classes=self.classification_param.num_classes,
            w_initializer=InitializerFactory.get_initializer(
                name=self.net_param.weight_initializer),
            b_initializer=InitializerFactory.get_initializer(
                name=self.net_param.bias_initializer),
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net_param.activation_function)

    def add_confusion_matrix_summaries_(self,
                                        outputs_collector,
                                        net_out,
                                        data_dict):
        """ This method defines several monitoring metrics that
        are derived from the confusion matrix """
        labels = tf.reshape(tf.cast(data_dict['label'], tf.int64), [-1])
        prediction = tf.reshape(tf.argmax(net_out, -1), [-1])
        num_classes = self.classification_param.num_classes
        conf_mat = tf.contrib.metrics.confusion_matrix(labels, prediction, num_classes)
        conf_mat = tf.to_float(conf_mat)
        if self.classification_param.num_classes == 2:
            outputs_collector.add_to_collection(
                var=conf_mat[1][1], name='true_positives',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=conf_mat[1][0], name='false_negatives',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=conf_mat[0][1], name='false_positives',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=conf_mat[0][0], name='true_negatives',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
        else:
            outputs_collector.add_to_collection(
                var=conf_mat[tf.newaxis, :, :, tf.newaxis],
                name='confusion_matrix',
                average_over_devices=True, summary_type='image',
                collection=TF_SUMMARIES)

        outputs_collector.add_to_collection(
            var=tf.trace(conf_mat), name='accuracy',
            average_over_devices=True, summary_type='scalar',
            collection=TF_SUMMARIES)


    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):

        def switch_sampler(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()

        if self.is_training:
            self.patience = self.action_param.patience
            self.mode = self.action_param.early_stopping_mode
            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(for_training=True),
                                    lambda: switch_sampler(for_training=False))
            else:
                data_dict = switch_sampler(for_training=True)

            image = tf.cast(data_dict['image'], tf.float32)
            net_args = {'is_training': self.is_training,
                        'keep_prob': self.net_param.keep_prob}
            net_out = self.net(image, **net_args)

            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)
            loss_func = LossFunction(
                n_class=self.classification_param.num_classes,
                loss_type=self.action_param.loss_type)
            data_loss = loss_func(
                prediction=net_out,
                ground_truth=data_dict.get('label', None))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                loss = data_loss + reg_loss
            else:
                loss = data_loss

            self.total_loss = loss

            grads = self.optimiser.compute_gradients(
                loss, colocate_gradients_with_ops=True)

            outputs_collector.add_to_collection(
                var=self.total_loss, name='total_loss',
                average_over_devices=True, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.total_loss, name='total_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables
            outputs_collector.add_to_collection(
                var=data_loss, name='data_loss',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss, name='data_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            self.add_confusion_matrix_summaries_(outputs_collector,
                                                 net_out,
                                                 data_dict)
        else:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            data_dict = switch_sampler(for_training=False)
            image = tf.cast(data_dict['image'], tf.float32)
            net_args = {'is_training': self.is_training,
                        'keep_prob': self.net_param.keep_prob}
            net_out = self.net(image, **net_args)
            tf.logging.info(
                'net_out.shape may need to be resized: %s', net_out.shape)
            output_prob = self.classification_param.output_prob
            num_classes = self.classification_param.num_classes
            if output_prob and num_classes > 1:
                post_process_layer = PostProcessingLayer(
                    'SOFTMAX', num_classes=num_classes)
            elif not output_prob and num_classes > 1:
                post_process_layer = PostProcessingLayer(
                    'ARGMAX', num_classes=num_classes)
            else:
                post_process_layer = PostProcessingLayer(
                    'IDENTITY', num_classes=num_classes)
            net_out = post_process_layer(net_out)

            outputs_collector.add_to_collection(
                var=net_out, name='window',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            self.initialise_aggregator()

    def interpret_output(self, batch_output):
        if not self.is_training:
            return self.output_decoder.decode_batch(
                {'csv': batch_output['window']},
                batch_output['location'])
        return True

    def initialise_evaluator(self, eval_param):
        self.eval_param = eval_param
        self.evaluator = ClassificationEvaluator(self.readers[0],
                                                 self.classification_param,
                                                 eval_param)

    def add_inferred_output(self, data_param, task_param):
        return self.add_inferred_output_like(data_param, task_param, 'label')
