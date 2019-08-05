import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.contrib.segmentation_selective_sampler.sampler_selective import \
    SelectiveSampler, Constraint
from niftynet.engine.application_factory import \
    ApplicationNetFactory, InitializerFactory, OptimiserFactory
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.loss_segmentation import LossFunction
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer

SUPPORTED_INPUT = set(['image', 'label', 'weight', 'sampler'])


class SelectiveSampling(BaseApplication):
    REQUIRED_CONFIG_SECTION = "SEGMENTATION"

    def __init__(self, net_param, action_param, is_training):
        super(SelectiveSampling, self).__init__()
        tf.logging.info('starting segmentation application')
        self.is_training = is_training

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.segmentation_param = None
        self.SUPPORTED_SAMPLING = {
            'selective': (self.initialise_selective_sampler,
                          self.initialise_grid_sampler,
                          self.initialise_grid_aggregator)
        }

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):

        self.data_param = data_param
        self.segmentation_param = task_param

        try:
            reader_phase = self.action_param.dataset_to_infer
        except AttributeError:
            reader_phase = None
        file_lists = data_partitioner.get_file_lists_by(
            phase=reader_phase, action=self.action)
        # read each line of csv files into an instance of Subject
        if self.is_training:
            self.readers = []
            for file_list in file_lists:
                reader = ImageReader(SUPPORTED_INPUT)
                reader.initialise(data_param, task_param, file_list)
                self.readers.append(reader)

        else:  # in the inference process use image input only
            inference_reader = ImageReader(['image'])
            inference_reader.initialise(data_param, task_param, file_lists[0])
            self.readers = [inference_reader]

        foreground_masking_layer = None
        if self.net_param.normalise_foreground_only:
            foreground_masking_layer = BinaryMaskingLayer(
                type_str=self.net_param.foreground_type,
                multimod_fusion=self.net_param.multimod_foreground_type,
                threshold=0.0)

        mean_var_normaliser = MeanVarNormalisationLayer(
            image_name='image', binary_masking_func=foreground_masking_layer)
        histogram_normaliser = None
        if self.net_param.histogram_ref_file:
            histogram_normaliser = HistogramNormalisationLayer(
                image_name='image',
                modalities=vars(task_param).get('image'),
                model_filename=self.net_param.histogram_ref_file,
                binary_masking_func=foreground_masking_layer,
                norm_type=self.net_param.norm_type,
                cutoff=self.net_param.cutoff,
                name='hist_norm_layer')

        label_normaliser = None
        if self.net_param.histogram_ref_file:
            label_normaliser = DiscreteLabelNormalisationLayer(
                image_name='label',
                modalities=vars(task_param).get('label'),
                model_filename=self.net_param.histogram_ref_file)

        normalisation_layers = []
        if self.net_param.normalisation:
            normalisation_layers.append(histogram_normaliser)
        if self.net_param.whitening:
            normalisation_layers.append(mean_var_normaliser)
        if task_param.label_normalisation:
            normalisation_layers.append(label_normaliser)

        augmentation_layers = []
        if self.is_training:
            if self.action_param.random_flipping_axes != -1:
                augmentation_layers.append(RandomFlipLayer(
                    flip_axes=self.action_param.random_flipping_axes))
            if self.action_param.scaling_percentage:
                augmentation_layers.append(RandomSpatialScalingLayer(
                    min_percentage=self.action_param.scaling_percentage[0],
                    max_percentage=self.action_param.scaling_percentage[1]))
            if self.action_param.rotation_angle or \
                    self.action_param.rotation_angle_x or \
                    self.action_param.rotation_angle_y or \
                    self.action_param.rotation_angle_z:
                rotation_layer = RandomRotationLayer()
                if self.action_param.rotation_angle:
                    rotation_layer.init_uniform_angle(
                        self.action_param.rotation_angle)
                else:
                    rotation_layer.init_non_uniform_angle(
                        self.action_param.rotation_angle_x,
                        self.action_param.rotation_angle_y,
                        self.action_param.rotation_angle_z)
                augmentation_layers.append(rotation_layer)

        volume_padding_layer = [PadLayer(
            image_name=SUPPORTED_INPUT,
            border=self.net_param.volume_padding_size,
            mode=self.net_param.volume_padding_mode,
            pad_to=self.net_param.volume_padding_to_size)
        ]

        for reader in self.readers:
            reader.add_preprocessing_layers(
                volume_padding_layer +
                normalisation_layers +
                augmentation_layers)

    def initialise_selective_sampler(self):
        # print("Initialisation ",
        #       self.segmentation_param.compulsory_labels,
        #       self.segmentation_param.proba_connect)
        # print(self.segmentation_param.num_min_labels,
        #       self.segmentation_param.proba_connect)
        selective_constraints = Constraint(
            self.segmentation_param.compulsory_labels,
            self.segmentation_param.min_sampling_ratio,
            self.segmentation_param.min_numb_labels,
            self.segmentation_param.proba_connect)
        self.sampler = [[
            SelectiveSampler(
                reader=reader,
                data_param=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=self.action_param.sample_per_volume,
                constraint=selective_constraints,
                random_windows_per_image=self.segmentation_param.rand_samples,
                queue_length=self.net_param.queue_length)
            for reader in self.readers]]

    def initialise_grid_sampler(self):
        self.sampler = [[GridSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            spatial_window_size=self.action_param.spatial_window_size,
            window_border=self.action_param.border,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_aggregator(self):
        self.output_decoder = GridSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order)

    def initialise_sampler(self):
        if self.is_training:
            self.SUPPORTED_SAMPLING['selective'][0]()
        else:
            self.SUPPORTED_SAMPLING['selective'][1]()

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
            num_classes=self.segmentation_param.num_classes,
            w_initializer=InitializerFactory.get_initializer(
                name=self.net_param.weight_initializer),
            b_initializer=InitializerFactory.get_initializer(
                name=self.net_param.bias_initializer),
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net_param.activation_function)

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
            net_out = self.net(image, is_training=self.is_training)

            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)
            loss_func = LossFunction(
                n_class=self.segmentation_param.num_classes,
                loss_type=self.action_param.loss_type)
            data_loss = loss_func(
                prediction=net_out,
                ground_truth=data_dict.get('label', None),
                weight_map=data_dict.get('weight', None))
            reg_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                loss = data_loss + reg_loss
            else:
                loss = data_loss
            grads = self.optimiser.compute_gradients(loss)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables
            outputs_collector.add_to_collection(
                var=data_loss, name='dice_loss',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss, name='dice_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            # outputs_collector.add_to_collection(
            #    var=image*180.0, name='image',
            #    average_over_devices=False, summary_type='image3_sagittal',
            #    collection=TF_SUMMARIES)

            # outputs_collector.add_to_collection(
            #    var=image, name='image',
            #    average_over_devices=False,
            #    collection=NETWORK_OUTPUT)

            # outputs_collector.add_to_collection(
            #    var=tf.reduce_mean(image), name='mean_image',
            #    average_over_devices=False, summary_type='scalar',
            #    collection=CONSOLE)
        else:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            data_dict = switch_sampler(for_training=False)
            image = tf.cast(data_dict['image'], tf.float32)
            net_out = self.net(image, is_training=self.is_training)

            output_prob = self.segmentation_param.output_prob
            num_classes = self.segmentation_param.num_classes
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
            init_aggregator = \
                self.SUPPORTED_SAMPLING[self.net_param.window_sampling][2]
            init_aggregator()

    def interpret_output(self, batch_output):
        if not self.is_training:
            return self.output_decoder.decode_batch(
                {'window_image': batch_output['window']},
                batch_output['location'])
        return True
