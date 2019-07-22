import tensorflow as tf
import copy

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import ApplicationNetFactory
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import NETWORK_OUTPUT
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.engine.sampler_weighted_v2 import WeightedSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.layer.crop import CropLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.loss_regression import LossFunction
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer

SUPPORTED_INPUT = set(['image', 'output', 'weight', 'sampler'])


class RegressionRecApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = "REGRESSION"

    def __init__(self, net_param, action_param, action):
        BaseApplication.__init__(self)
        tf.logging.info('starting recursive regression application')
        self.action = action

        self.net_param = net_param
        self.net2_param = copy.deepcopy(net_param)
        self.action_param = action_param
        self.regression_param = None

        self.data_param = None
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
        }

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        self.data_param = data_param
        self.regression_param = task_param

        # read each line of csv files into an instance of Subject
        if self.is_training:
            file_lists = []
            if self.action_param.validation_every_n > 0:
                file_lists.append(data_partitioner.train_files)
                file_lists.append(data_partitioner.validation_files)
            else:
                file_lists.append(data_partitioner.train_files)

            self.readers = []
            for file_list in file_lists:
                reader = ImageReader(SUPPORTED_INPUT)
                reader.initialise(data_param, task_param, file_list)
                self.readers.append(reader)
        else:
            inference_reader = ImageReader(['image'])
            file_list = data_partitioner.inference_files
            inference_reader.initialise(data_param, task_param, file_list)
            self.readers = [inference_reader]

        mean_var_normaliser = MeanVarNormalisationLayer(
            image_name='image')
        histogram_normaliser = None
        if self.net_param.histogram_ref_file:
            histogram_normaliser = HistogramNormalisationLayer(
                image_name='image',
                modalities=vars(task_param).get('image'),
                model_filename=self.net_param.histogram_ref_file,
                norm_type=self.net_param.norm_type,
                cutoff=self.net_param.cutoff,
                name='hist_norm_layer')

        normalisation_layers = []
        if self.net_param.normalisation:
            normalisation_layers.append(histogram_normaliser)
        if self.net_param.whitening:
            normalisation_layers.append(mean_var_normaliser)

        augmentation_layers = []
        if self.is_training:
            if self.action_param.random_flipping_axes != -1:
                augmentation_layers.append(RandomFlipLayer(
                    flip_axes=self.action_param.random_flipping_axes))
            if self.action_param.scaling_percentage:
                augmentation_layers.append(RandomSpatialScalingLayer(
                    min_percentage=self.action_param.scaling_percentage[0],
                    max_percentage=self.action_param.scaling_percentage[1]))
            if self.action_param.rotation_angle:
                augmentation_layers.append(RandomRotationLayer())
                augmentation_layers[-1].init_uniform_angle(
                    self.action_param.rotation_angle)

        volume_padding_layer = []
        if self.net_param.volume_padding_size:
            volume_padding_layer.append(PadLayer(
                image_name=SUPPORTED_INPUT,
                border=self.net_param.volume_padding_size))
        for reader in self.readers:
            reader.add_preprocessing_layers(volume_padding_layer +
                                            normalisation_layers +
                                            augmentation_layers)

    def initialise_uniform_sampler(self):
        self.sampler = [[UniformSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_weighted_sampler(self):
        self.sampler = [[WeightedSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_resize_sampler(self):
        self.sampler = [[ResizeSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            shuffle=self.is_training,
            smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_sampler(self):
        self.sampler = [[GridSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            spatial_window_size=self.action_param.spatial_window_size,
            window_border=self.action_param.border,
            smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_balanced_sampler(self):
        self.sampler = [[BalancedSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_aggregator(self):
        self.output_decoder = GridSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order)

    def initialise_resize_aggregator(self):
        self.output_decoder = ResizeSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order)

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
            num_classes=1,
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net_param.activation_function)

        self.net2 = ApplicationNetFactory.create(self.net2_param.name)(
            num_classes=1,
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net2_param.activation_function)

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
                                    lambda: switch_sampler(True),
                                    lambda: switch_sampler(False))
            else:
                data_dict = switch_sampler(for_training=True)

            image = tf.cast(data_dict['image'], tf.float32)
            pct1_out = self.net(image, self.is_training)
            res2_out = self.net2(tf.concat([image, pct1_out],4), self.is_training)
            pct2_out = tf.add(pct1_out,res2_out)
            res3_out = self.net2(tf.concat([image, pct2_out],4), self.is_training)
            pct3_out = tf.add(pct2_out,res3_out)
            #res4_out = self.net2(tf.concat([image, pct3_out],4), self.is_training)
            #pct4_out = tf.add(pct3_out,res4_out)
	    #net_out = self.net(image, is_training=self.is_training)
            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)
            loss_func = LossFunction(
                loss_type=self.action_param.loss_type)

            crop_layer = CropLayer(
                border=self.regression_param.loss_border, name='crop-88')

            data_loss1 = loss_func(
                prediction=crop_layer(pct1_out),
                ground_truth=crop_layer(data_dict.get('output', None)),
                weight_map=None if data_dict.get('weight', None) is None else crop_layer(data_dict.get('weight', None)))
            data_loss2 = loss_func(
                prediction=crop_layer(pct2_out),
                ground_truth=crop_layer(data_dict.get('output', None)),
                weight_map=None if data_dict.get('weight', None) is None else crop_layer(data_dict.get('weight', None)))
            data_loss3 = loss_func(
                prediction=crop_layer(pct3_out),
                ground_truth=crop_layer(data_dict.get('output', None)),
                weight_map=None if data_dict.get('weight', None) is None else crop_layer(data_dict.get('weight', None)))

            #prediction = crop_layer(net_out)
            #ground_truth = crop_layer(data_dict.get('output', None))
            #weight_map = None if data_dict.get('weight', None) is None \
                #else crop_layer(data_dict.get('weight', None))
            #data_loss = loss_func(prediction=prediction,
                                  #ground_truth=ground_truth,
                                  #weight_map=weight_map)

            reg_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                loss = reg_loss + data_loss1 + data_loss2 + data_loss3
            else:
                loss = data_loss1 + data_loss2 + data_loss3
            grads = self.optimiser.compute_gradients(loss)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables
            outputs_collector.add_to_collection(
                var=loss, name='Loss',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss1, name='data_loss1',
                average_over_devices=True,
                collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss2, name='data_loss2',
                average_over_devices=True,
                collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss3, name='data_loss3',
                average_over_devices=True,
                collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss1, name='data_loss1',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=data_loss2, name='data_loss2',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=data_loss3, name='data_loss3',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=loss, name='LossSum',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
#            outputs_collector.add_to_collection(
#                var=pct3_out, name="pct3_out",
#                average_over_devices=True, summary_type="image3_axial",
#                collection=TF_SUMMARIES)

        else:
            data_dict = switch_sampler(for_training=False)
            image = tf.cast(data_dict['image'], tf.float32)
            #net_out = self.net(image, is_training=self.is_training)
            pct1_out = self.net(image, self.is_training)
            res2_out = self.net2(tf.concat([image, pct1_out],4), self.is_training)
            pct2_out = tf.add(pct1_out,res2_out)
            res3_out = self.net2(tf.concat([image, pct2_out],4), self.is_training)
            pct3_out = tf.add(pct2_out,res3_out)
            res4_out = self.net2(tf.concat([image, pct3_out],4), self.is_training)
            pct4_out = tf.add(pct3_out,res4_out)
            crop_layer = CropLayer(border=0, name='crop-88')
            post_process_layer = PostProcessingLayer('IDENTITY')
            net_out = post_process_layer(crop_layer(pct4_out))

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
        else:
            return True
