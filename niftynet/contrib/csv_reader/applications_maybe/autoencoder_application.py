import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import ApplicationNetFactory
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import NETWORK_OUTPUT
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.sampler_linear_interpolate_v2 import LinearInterpolateSampler
from niftynet.contrib.csv_reader.sampler_linear_interpolate_v2_csv import \
    LinearInterpolateSamplerCSV as LinearInterpolateSampler
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.contrib.csv_reader.sampler_resize_v2_csv import \
    ResizeSamplerCSV as ResizeSampler
from niftynet.contrib.csv_reader.csv_reader import CSVReader
from niftynet.engine.windows_aggregator_identity import WindowAsImageAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.layer.loss_autoencoder import LossFunction
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_INPUT = set(['image', 'feature'])
SUPPORTED_INFERENCE = \
    set(['encode', 'encode-decode', 'sample', 'linear_interpolation'])


class AutoencoderApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = "AUTOENCODER"

    def __init__(self, net_param, action_param, action):
        BaseApplication.__init__(self)
        tf.logging.info('starting autoencoder application')

        self.action = action

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.autoencoder_param = None

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        self.data_param = data_param
        self.autoencoder_param = task_param

        if not self.is_training:
            self._infer_type = look_up_operations(
                self.autoencoder_param.inference_type, SUPPORTED_INFERENCE)
        else:
            self._infer_type = None
        try:
            reader_phase = self.action_param.dataset_to_infer
        except AttributeError:
            reader_phase = None
        file_lists = data_partitioner.get_file_lists_by(
            phase=reader_phase, action=self.action)
        # read each line of csv files into an instance of Subject
        if self.is_evaluation:
            NotImplementedError('Evaluation is not yet '
                                'supported in this application.')
        if self.is_training:
            self.readers = []
            self.csv_reader = []
            for file_list in file_lists:
                reader = ImageReader(['image'])
                reader.initialise(data_param, task_param, file_list)
                self.readers.append(reader)
        if self._infer_type in ('encode', 'encode-decode'):
            self.readers = [ImageReader(['image'])]
            self.readers[0].initialise(data_param, task_param, file_lists[0])
        elif self._infer_type == 'sample':
            self.readers = []
            self.csv_reader = []
        elif self._infer_type == 'linear_interpolation':
            self.csv_reader = []
            self.readers = [ImageReader(['feature'])]
            self.readers[0].initialise(data_param, task_param, file_lists[0])
        # if self.is_training or self._infer_type in ('encode', 'encode-decode'):
        #    mean_var_normaliser = MeanVarNormalisationLayer(image_name='image')
        #    self.reader.add_preprocessing_layers([mean_var_normaliser])

    def initialise_sampler(self):
        self.sampler = []
        if self.is_training:
            self.sampler.append([ResizeSampler(
                reader=reader,
                csv_reader=None,
                window_sizes=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=1,
                shuffle=True,
                queue_length=self.net_param.queue_length) for reader in
                self.readers])
            return
        if self._infer_type in ('encode', 'encode-decode'):
            self.sampler.append([ResizeSampler(
                reader=reader,
                csv_reader=None,
                window_sizes=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=1,
                shuffle=False,
                queue_length=self.net_param.queue_length) for reader in
                self.readers])
            return
        if self._infer_type == 'linear_interpolation':
            self.sampler.append([LinearInterpolateSampler(
                reader=reader,
                csv_reader=None,
                window_sizes=self.data_param,
                batch_size=self.net_param.batch_size,
                n_interpolations=self.autoencoder_param.n_interpolations,
                queue_length=self.net_param.queue_length) for reader in
                self.readers])
            return

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
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        def switch_sampler(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()

        if self.is_training:
            self.patience = self.action_param.patience
            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(True),
                                    lambda: switch_sampler(False))
            else:
                data_dict = switch_sampler(for_training=True)

            image = tf.cast(data_dict['image'], tf.float32)
            net_output = self.net(image, is_training=self.is_training)

            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)

            loss_func = LossFunction(loss_type=self.action_param.loss_type)
            data_loss = loss_func(net_output)
            loss = data_loss
            if self.net_param.decay > 0.0:
                reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                if reg_losses:
                    reg_loss = tf.reduce_mean(
                        [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                    loss = loss + reg_loss

            self.total_loss = loss
            grads = self.optimiser.compute_gradients(
                loss, colocate_gradients_with_ops=True)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])

            outputs_collector.add_to_collection(
                var=self.total_loss, name='total_loss',
                average_over_devices=True, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.total_loss, name='total_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=data_loss, name='variational_lower_bound',
                average_over_devices=True, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss, name='variational_lower_bound',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=net_output[4], name='Originals',
                average_over_devices=False, summary_type='image3_coronal',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=net_output[2], name='Means',
                average_over_devices=False, summary_type='image3_coronal',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=net_output[5], name='Variances',
                average_over_devices=False, summary_type='image3_coronal',
                collection=TF_SUMMARIES)
        else:
            if self._infer_type in ('encode', 'encode-decode'):
                data_dict = self.get_sampler()[0][0].pop_batch_op()
                image = tf.cast(data_dict['image'], dtype=tf.float32)
                net_output = self.net(image, is_training=False)

                outputs_collector.add_to_collection(
                    var=data_dict['image_location'], name='location',
                    average_over_devices=True, collection=NETWORK_OUTPUT)

                if self._infer_type == 'encode-decode':
                    outputs_collector.add_to_collection(
                        var=net_output[2], name='generated_image',
                        average_over_devices=True, collection=NETWORK_OUTPUT)
                if self._infer_type == 'encode':
                    outputs_collector.add_to_collection(
                        var=net_output[7], name='embedded',
                        average_over_devices=True, collection=NETWORK_OUTPUT)

                self.output_decoder = WindowAsImageAggregator(
                    image_reader=self.readers[0],
                    output_path=self.action_param.save_seg_dir)
                return
            elif self._infer_type == 'sample':
                image_size = (self.net_param.batch_size,) + \
                             self.action_param.spatial_window_size + (1,)
                dummy_image = tf.zeros(image_size)
                net_output = self.net(dummy_image, is_training=False)
                noise_shape = net_output[-1].shape.as_list()
                stddev = self.autoencoder_param.noise_stddev
                noise = tf.random_normal(shape=noise_shape,
                                         mean=0.0,
                                         stddev=stddev,
                                         dtype=tf.float32)
                partially_decoded_sample = self.net.shared_decoder(
                    noise, is_training=False)
                decoder_output = self.net.decoder_means(
                    partially_decoded_sample, is_training=False)

                outputs_collector.add_to_collection(
                    var=decoder_output, name='generated_image',
                    average_over_devices=True, collection=NETWORK_OUTPUT)
                self.output_decoder = WindowAsImageAggregator(
                    image_reader=None,
                    output_path=self.action_param.save_seg_dir)
                return
            elif self._infer_type == 'linear_interpolation':
                # construct the entire network
                image_size = (self.net_param.batch_size,) + \
                             self.action_param.spatial_window_size + (1,)
                dummy_image = tf.zeros(image_size)
                net_output = self.net(dummy_image, is_training=False)
                data_dict = self.get_sampler()[0][0].pop_batch_op()
                real_code = data_dict['feature']
                real_code = tf.reshape(real_code, net_output[-1].get_shape())
                partially_decoded_sample = self.net.shared_decoder(
                    real_code, is_training=False)
                decoder_output = self.net.decoder_means(
                    partially_decoded_sample, is_training=False)

                outputs_collector.add_to_collection(
                    var=decoder_output, name='generated_image',
                    average_over_devices=True, collection=NETWORK_OUTPUT)
                outputs_collector.add_to_collection(
                    var=data_dict['feature_location'], name='location',
                    average_over_devices=True, collection=NETWORK_OUTPUT)
                self.output_decoder = WindowAsImageAggregator(
                    image_reader=self.readers[0],
                    output_path=self.action_param.save_seg_dir)
            else:
                raise NotImplementedError

    def interpret_output(self, batch_output):
        if self.is_training:
            return True
        else:
            infer_type = look_up_operations(
                self.autoencoder_param.inference_type,
                SUPPORTED_INFERENCE)
            if infer_type == 'encode':
                return self.output_decoder.decode_batch(
                    {'window_embedded':batch_output['embedded']},
                    batch_output['location'][:, 0:1])
            if infer_type == 'encode-decode':
                return self.output_decoder.decode_batch(
                    {'window_generated_image':batch_output['generated_image']},
                    batch_output['location'][:, 0:1])
            if infer_type == 'sample':
                return self.output_decoder.decode_batch(
                    {'generated_image':batch_output['generated_image']},
                    None)
            if infer_type == 'linear_interpolation':
                return self.output_decoder.decode_batch(
                    {'generated_image':batch_output['generated_image']},
                    batch_output['location'][:, :2])
