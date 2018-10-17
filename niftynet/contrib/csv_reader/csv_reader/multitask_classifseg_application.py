# -*- coding: utf-8 -*-
"""
This module defines an image-level classification application
that maps from images to scalar, multi-class labels.

This class is instantiated and initalized by the application_driver.
"""

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import \
    ApplicationNetFactory, InitializerFactory, OptimiserFactory
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.contrib.csv_reader.sampler_resize_v2_csv import ResizeSamplerCSV \
    as ResizeSampler
from niftynet.contrib.csv_reader.sampler_uniform_v2_csv import \
    UniformSamplerCSV as UniformSampler
from niftynet.contrib.csv_reader.sampler_weighted_v2_csv import \
    WeightedSamplerCSV as WeightedSampler
from niftynet.contrib.csv_reader.sampler_balanced_v2_csv import \
    BalancedSamplerCSV as BalancedSampler
from niftynet.contrib.csv_reader.sampler_grid_v2_csv import GridSamplerCSV as\
    GridSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.contrib.csv_reader.csv_reader import CSVReader
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.layer.loss_classification import LossFunction as \
    LossFunctionClassification
from niftynet.layer.loss_segmentation import \
    LossFunction as LossFunctionSegmentation
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.evaluation.classification_evaluator import ClassificationEvaluator

SUPPORTED_INPUT = set(['image', 'value', 'label', 'sampler', 'inferred'])


class MultiClassifSegApplication(BaseApplication):
    """This class defines an application for image-level classification
    problems mapping from images to scalar labels.

    This is the application class to be instantiated by the driver
    and referred to in configuration files.

    Although structurally similar to segmentation, this application
    supports different samplers/aggregators (because patch-based
    processing is not appropriate), and monitoring metrics."""

    REQUIRED_CONFIG_SECTION = "SEGMENTATION"

    def __init__(self, net_param, action_param, action):
        super(MultiClassifSegApplication, self).__init__()
        tf.logging.info('starting classification application')
        self.action = action

        self.net_param = net_param
        self.eval_param = None
        self.evaluator = None
        self.action_param = action_param
        self.net_multi = None
        self.data_param = None
        self.segmentation_param = None
        self.csv_readers = None
        self.SUPPORTED_SAMPLING = {
            'uniform': (self.initialise_uniform_sampler,
                        self.initialise_grid_sampler,
                        self.initialise_grid_aggregator),
            'weighted': (self.initialise_weighted_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
            'resize': (self.initialise_resize_sampler,
                       self.initialise_resize_sampler,
                       self.initialise_resize_aggregator),
            'balanced': (self.initialise_balanced_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
        }

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        '''
        Initialise the data loader both csv readers and image readers and
        specify preprocessing layers
        :param data_param:
        :param task_param:
        :param data_partitioner:
        :return:
        '''

        self.data_param = data_param
        self.segmentation_param = task_param

        if self.is_training:
            image_reader_names = ('image', 'sampler', 'label')
            csv_reader_names = ('value',)
        elif self.is_inference:
            image_reader_names = ('image',)
            csv_reader_names = ()
        elif self.is_evaluation:
            image_reader_names = ('image', 'inferred', 'label')
            csv_reader_names = ('value',)
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
            ImageReader(image_reader_names).initialise(
                data_param, task_param, file_list) for file_list in file_lists]
        if self.is_inference:
            self.action_param.sample_per_volume = 1
        if csv_reader_names is not None and list(csv_reader_names):
            self.csv_readers = [
                CSVReader(csv_reader_names).initialise(
                    data_param, task_param, file_list,
                    sample_per_volume=self.action_param.sample_per_volume)
                for file_list in file_lists]
        else:
            self.csv_readers = [None for file_list in file_lists]

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
                    max_percentage=train_param.scaling_percentage[1]))
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

    def initialise_uniform_sampler(self):
        '''
        Create the uniform sampler using information from readers
        :return:
        '''
        self.sampler = [[UniformSampler(
            reader=reader,
            csv_reader=csv_reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader, csv_reader in
                         zip(self.readers, self.csv_readers)]]

    def initialise_weighted_sampler(self):
        '''
        Create the weighted sampler using the info from the csv_readers and
        image_readers and the configuration parameters
        :return:
        '''
        self.sampler = [[WeightedSampler(
            reader=reader,
            csv_reader=csv_reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader, csv_reader in
                         zip(self.readers, self.csv_readers)]]

    def initialise_resize_sampler(self):
        '''
        Define the resize sampler using the information from the
        configuration parameters, csv_readers and image_readers
        :return:
        '''
        self.sampler = [[ResizeSampler(
            reader=reader,
            csv_reader=csv_reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            shuffle=self.is_training,
            smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
            queue_length=self.net_param.queue_length) for reader, csv_reader in
                         zip(self.readers, self.csv_readers)]]

    def initialise_grid_sampler(self):
        '''
        Define the grid sampler based on the information from configuration
        and the csv_readers and image_readers specifications
        :return:
        '''
        self.sampler = [[GridSampler(
            reader=reader,
            csv_reader=csv_reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            spatial_window_size=self.action_param.spatial_window_size,
            window_border=self.action_param.border,
            smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
            queue_length=self.net_param.queue_length) for reader, csv_reader in
                         zip(self.readers, self.csv_readers)]]

    def initialise_balanced_sampler(self):
        '''
        Define the balanced sampler based on the information from configuration
        and the csv_readers and image_readers specifications
        :return:
        '''
        self.sampler = [[BalancedSampler(
            reader=reader,
            csv_reader=csv_reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader, csv_reader in
                         zip(self.readers, self.csv_readers)]]

    def initialise_grid_aggregator(self):
        '''
        Define the grid aggregator used for decoding using configuration
        parameters
        :return:
        '''
        self.output_decoder = GridSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order,
            postfix=self.action_param.output_postfix,
            fill_constant=self.action_param.fill_constant)

    def initialise_resize_aggregator(self):
        '''
        Define the resize aggregator used for decoding using the
        configuration parameters
        :return:
        '''
        self.output_decoder = ResizeSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order,
            postfix=self.action_param.output_postfix)

    def initialise_sampler(self):
        '''
        Specifies the sampler used among those previously defined based on
        the sampling choice
        :return:
        '''
        if self.is_training:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][0]()
        elif self.is_inference:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][1]()

    def initialise_aggregator(self):
        '''
        Specifies the aggregator used based on the sampling choice
        :return:
        '''
        self.SUPPORTED_SAMPLING[self.net_param.window_sampling][2]()

    def initialise_network(self):
        '''
        Initialise the network and specifies the ordering of elements
        :return:
        '''
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

        self.net = ApplicationNetFactory.create(
            'niftynet.contrib.csv_reader.toynet_features.ToyNetFeat')(
                num_classes=self.segmentation_param.num_classes,
                w_initializer=InitializerFactory.get_initializer(
                    name=self.net_param.weight_initializer),
                b_initializer=InitializerFactory.get_initializer(
                    name=self.net_param.bias_initializer),
                w_regularizer=w_regularizer,
                b_regularizer=b_regularizer,
                acti_func=self.net_param.activation_function)
        self.net_multi = ApplicationNetFactory.create(
            'niftynet.contrib.csv_reader.class_seg_finnet.ClassSegFinnet')(
                num_classes=self.segmentation_param.num_classes,
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
        num_classes = 2
        conf_mat = tf.confusion_matrix(labels, prediction, num_classes)
        conf_mat = tf.to_float(conf_mat)
        if self.segmentation_param.num_classes == 2:
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
            net_out_seg, net_out_class = self.net_multi(net_out,
                                                        self.is_training)

            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)
            loss_func_class = LossFunctionClassification(
                n_class=2,
                loss_type='CrossEntropy')
            loss_func_seg = LossFunctionSegmentation(
                n_class=self.segmentation_param.num_classes,
                loss_type=self.action_param.loss_type)
            data_loss_seg = loss_func_seg(
                prediction=net_out_seg,
                ground_truth=data_dict.get('label', None))
            data_loss_class = loss_func_class(
                prediction=net_out_class,
                ground_truth=data_dict.get('value', None))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                loss = data_loss_seg + data_loss_class + reg_loss
            else:
                loss = data_loss_seg + data_loss_class
            self.total_loss = loss
            self.total_loss = tf.Print(tf.cast(self.total_loss, tf.float32),
                                       [loss, tf.shape(net_out_seg),
                                        tf.shape(net_out_class)],
                                       message='test')
            grads = self.optimiser.compute_gradients(
                loss, colocate_gradients_with_ops=True)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables
            outputs_collector.add_to_collection(
                var=data_loss_class, name='data_loss',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss_seg, name='data_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            # self.add_confusion_matrix_summaries_(outputs_collector,
            #                                      net_out_class,
            #                                      data_dict)
        else:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            data_dict = switch_sampler(for_training=False)
            image = tf.cast(data_dict['image'], tf.float32)
            net_args = {'is_training': self.is_training,
                        'keep_prob': self.net_param.keep_prob}
            net_out = self.net(image, **net_args)
            net_out_seg, net_out_class = self.net_multi(net_out,
                                                        self.is_training)
            tf.logging.info(
                'net_out.shape may need to be resized: %s', net_out.shape)
            output_prob = self.segmentation_param.output_prob
            num_classes = self.segmentation_param.num_classes
            if output_prob and num_classes > 1:
                post_process_layer_class = PostProcessingLayer(
                    'SOFTMAX', num_classes=num_classes)
                post_process_layer_seg = PostProcessingLayer('SOFTMAX',
                                                             num_classes=2)
            elif not output_prob and num_classes > 1:
                post_process_layer_class = PostProcessingLayer(
                    'ARGMAX', num_classes=num_classes)
                post_process_layer_seg = PostProcessingLayer('ARGMAX',
                                                             num_classes=2)
            else:
                post_process_layer_class = PostProcessingLayer(
                    'IDENTITY', num_classes=num_classes)
                post_process_layer_seg = PostProcessingLayer('IDENTITY',
                                                             num_classes=2)

            net_out_class = post_process_layer_class(net_out_class)
            net_out_seg = post_process_layer_seg(net_out_seg)

            outputs_collector.add_to_collection(
                var=net_out_seg, name='seg',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(var=net_out_class,
                                                name='value',
                                                average_over_devices=False,
                                                collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            self.initialise_aggregator()

    def interpret_output(self, batch_output):
        '''
        Specifies how the output should be decoded
        :param batch_output:
        :return:
        '''
        if not self.is_training:
            return self.output_decoder.decode_batch(
                {'window_seg': batch_output['seg'],
                 'csv_class': batch_output['value']},
                batch_output['location'])
        return True

    def initialise_evaluator(self, eval_param):
        '''
        Define the evaluator
        :param eval_param:
        :return:
        '''
        self.eval_param = eval_param
        self.evaluator = ClassificationEvaluator(self.readers[0],
                                                 self.segmentation_param,
                                                 eval_param)

    def add_inferred_output(self, data_param, task_param):
        '''
        Define how to treat added inferred output
        :param data_param:
        :param task_param:
        :return:
        '''
        return self.add_inferred_output_like(data_param, task_param, 'label')
