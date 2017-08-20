import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.sampler_resize import ResizeSampler
from niftynet.io.image_reader import ImageReader
from niftynet.layer.loss_autoencoder import LossFunction
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.utilities.util_common import look_up_operations
from niftynet.engine.image_windows_aggregator import WindowAsImageAggregator


SUPPORTED_INPUT = {'image', 'feature'}
SUPPORTED_INFERENCE = {
    'encode', 'encode-decode', 'sample', 'linear_interpolation'}


class AutoencoderApplication(BaseApplication):
    def __init__(self, net_param, action_param, is_training):
        BaseApplication.__init__(self)
        tf.logging.info('starting autoencoder application')

        self.is_training = is_training

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.autoencoder_param = None

    def initialise_dataset_loader(self, data_param=None, task_param=None):
        self.data_param = data_param
        self.autoencoder_param = task_param

        # read each line of csv files into an instance of Subject
        if self.is_training:
            self.reader = ImageReader(SUPPORTED_INPUT)
        else:  # in the inference process use image input only
            self.reader = ImageReader(['image'])
        self.reader.initialise_reader(data_param, task_param)

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
                augmentation_layers.append(RandomRotationLayer(
                    min_angle=self.action_param.rotation_angle[0],
                    max_angle=self.action_param.rotation_angle[1]))
        self.reader.add_preprocessing_layers(augmentation_layers)

    def initialise_sampler(self):
        self.sampler = []
        if self.is_training:
            self.sampler.append(ResizeSampler(
                reader=self.reader,
                data_param=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=1,
                shuffle_buffer=True))
        else:
            infer_type = look_up_operations(
                self.autoencoder_param.inference_type,
                SUPPORTED_INFERENCE)
            if infer_type in ('encode', 'encode-decode'):
                self.sampler.append(ResizeSampler(
                    reader=self.reader,
                    data_param=self.data_param,
                    batch_size=self.net_param.batch_size,
                    windows_per_image=1,
                    shuffle_buffer=False))

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

        self.net = AutoencoderFactory.create(self.net_param.name)(
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        if self.is_training:

            with tf.name_scope('Optimiser'):
                self.optimiser = tf.train.AdamOptimizer(
                    learning_rate=self.action_param.lr)
            device_id = gradients_collector.current_tower_id
            data_dict = self.get_sampler()[0].pop_batch_op(device_id)
            image = tf.cast(data_dict['image'], dtype=tf.float32)
            net_output = self.net(image, is_training=True)
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
            grads = self.optimiser.compute_gradients(loss)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])

            outputs_collector.add_to_collection(
                var=data_loss, name='variational_lower_bound',
                average_over_devices=True, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss, name='variational_lower_bound',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
        else:
            infer_type = look_up_operations(
                self.autoencoder_param.inference_type,
                SUPPORTED_INFERENCE)
            if infer_type in ('encode', 'encode-decode'):
                data_dict = self.get_sampler()[0].pop_batch_op()
                image = tf.cast(data_dict['image'], dtype=tf.float32)
                net_output = self.net(image, is_training=False)

                outputs_collector.add_to_collection(
                    var=data_dict['image_location'], name='location',
                    average_over_devices=True, collection=CONSOLE)

                if infer_type == 'encode-decode':
                    outputs_collector.add_to_collection(
                        var=net_output[2], name='generated_image',
                        average_over_devices=True, collection=CONSOLE)
                if infer_type == 'encode':
                    outputs_collector.add_to_collection(
                        var=net_output[7], name='embedded',
                        average_over_devices=True, collection=CONSOLE)

                self.output_decoder = WindowAsImageAggregator(
                    image_reader=self.reader,
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
                    batch_output['embedded'], batch_output['location'])
            if infer_type == 'encode-decode':
                return self.output_decoder.decode_batch(
                    batch_output['generated_image'], batch_output['location'])


class AutoencoderFactory(object):
    @staticmethod
    def create(name):
        if name == "vae":
            from niftynet.network.vae import VAE
            return VAE
        else:
            tf.logging.fatal("network: \"{}\" not implemented".format(name))
            raise NotImplementedError
