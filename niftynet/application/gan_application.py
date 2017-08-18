from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.image_windows_aggregator import BatchSplitingAggregator
from niftynet.engine.sampler_random_vector import RandomVectorSampler
from niftynet.engine.sampler_resize import ResizeSampler
from niftynet.io.image_reader import ImageReader
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.gan_loss import LossFunction
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer

SUPPORTED_INPUT = {'image', 'conditioning'}


class GanFactory(object):
    @staticmethod
    def create(name):
        if name == "simulator_gan":
            from niftynet.network.simulator_gan import SimulatorGAN
            return SimulatorGAN
        if name == "simple_gan":
            from niftynet.network.simple_gan import SimpleGAN
            return SimpleGAN
        else:
            print("network: \"{}\" not implemented".format(name))
            raise NotImplementedError


class GANApplication(BaseApplication):
    # def __init__(self, param):
    #     # super(GANApplication, self).__init__()
    #     self._net_class = None
    #     self._param = param
    #     self._volume_loader = None
    #     self._loss_func = LossFunction(loss_type=self._param.loss_type)
    #     self.num_objectives = 2
    #     self._net = None
    #     self._net_class_module = NetFactory.create(param.net_name)

    def set_model_param(self, net_param, action_param, is_training):
        self.is_training = is_training
        self.net_param = net_param
        self.action_param = action_param

        self.reader = None
        self.data_param = None
        self.gan_param = None

    def initialise_dataset_loader(self, data_param, gan_param):
        self.data_param = data_param
        self.gan_param = gan_param

        # read each line of csv files into an instance of Subject
        if self.is_training:
            self.reader = ImageReader(['image', 'conditioning'])
        else:  # in the inference process use image input only
            self.reader = ImageReader(['conditioning'])
        if self.reader:
            self.reader.initialise_reader(data_param, gan_param)

        if self.net_param.normalise_foreground_only:
            foreground_masking_layer = BinaryMaskingLayer(
                type=self.net_param.foreground_type,
                multimod_fusion=self.net_param.multimod_foreground_type,
                threshold=0.0)
        else:
            foreground_masking_layer = None

        mean_var_normaliser = MeanVarNormalisationLayer(
            field='image',
            binary_masking_func=foreground_masking_layer)
        if self.net_param.histogram_ref_file:
            histogram_normaliser = HistogramNormalisationLayer(
                field='image',
                modalities=vars(gan_param).get('image'),
                model_filename=self.net_param.histogram_ref_file,
                binary_masking_func=foreground_masking_layer,
                norm_type=self.net_param.norm_type,
                cutoff=self.net_param.cutoff,
                name='hist_norm_layer')
        else:
            histogram_normaliser = None

        normalisation_layers = []
        if self.net_param.normalisation:
            normalisation_layers.append(histogram_normaliser)
        if self.net_param.whitening:
            normalisation_layers.append(mean_var_normaliser)

        augmentation_layers = []
        if self.is_training:
            if self.action_param.random_flipping_axes > 0:
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

        if self.reader:
            self.reader.add_preprocessing_layers(
                normalisation_layers + augmentation_layers)

    def inference_sampler(self):
        pass

    def get_sampler(self):
        return self._sampler

    def initialise_sampler(self, is_training):
        self._sampler = []
        if is_training:
            self._sampler.append(ResizeSampler(
                reader=self.reader,
                data_param=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=1,
                shuffle_buffer=True))
        else:
            self._sampler.append(RandomVectorSampler(
                names=('vector',),
                vector_size=(self.gan_param.noise_size,),
                batch_size=self.net_param.batch_size,
                n_interpolations=self.gan_param.n_interpolations,
                repeat=None))
            self._sampler.append(ResizeSampler(
                reader=self.reader,
                data_param=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=self.gan_param.n_interpolations,
                shuffle_buffer=False))

    def training_sampler(self):
        pass

    def initialise_network(self):
        self._net = GanFactory.create(self.net_param.name)()

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 training_grads_collector=None):
        if self.is_training:
            with tf.name_scope('Optimizer'):
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.action_param.lr, beta1=0.5)

            # a new pop_batch_op for each gpu tower
            device_id = training_grads_collector.current_tower_id
            data_dict = self.get_sampler()[0].pop_batch_op(device_id)

            images = data_dict['image']
            noise_shape = [self.net_param.batch_size,
                           self.gan_param.noise_size]
            noise = tf.Variable(tf.random_normal(shape=noise_shape,
                                                 mean=0.0,
                                                 stddev=1.0,
                                                 dtype=tf.float32))
            tf.stop_gradient(noise)
            conditioning = data_dict['conditioning']
            net_output = self._net(noise,
                                   images,
                                   conditioning,
                                   self.is_training)

            self._loss_func = LossFunction(
                loss_type=self.action_param.loss_type)
            lossG, lossD = self.loss_func(net_output)
            if self.net_param.decay > 0:
                reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                if reg_losses:
                    reg_loss = tf.reduce_mean(
                        [tf.reduce_mean(l_reg) for l_reg in reg_losses])
                    lossD = lossD + reg_loss
                    lossG = lossG + reg_loss

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

            with tf.name_scope('ComputeGradients'):
                # gradients of generator
                generator_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                generator_grads = self.optimizer.compute_gradients(
                    lossG, var_list=generator_variables)

                # gradients of discriminator
                discriminator_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                discriminator_grads = self.optimizer.compute_gradients(
                    lossD, var_list=discriminator_variables)
                grads = [generator_grads, discriminator_grads]

                # add the grads back to application_driver's training_grads
                training_grads_collector.add_to_collection(grads)
            return net_output
        else:
            data_dict = self.get_sampler()[0].pop_batch_op()
            conditioning_dict = self.get_sampler()[1].pop_batch_op()
            conditioning = conditioning_dict['conditioning']
            image_size = conditioning.shape.as_list()[:-1]
            dummy_image = tf.zeros(image_size + [1])
            net_output = self._net(data_dict['vector'],
                                   dummy_image,
                                   conditioning,
                                   self.is_training)
            outputs_collector.add_to_collection(
                var=net_output[0],
                name='image',
                average_over_devices=False,
                collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=conditioning_dict['conditioning_location'],
                name='location',
                average_over_devices=False, collection=CONSOLE)

            self.output_decoder = BatchSplitingAggregator(
                image_reader=self.reader,
                output_path=self.action_param.save_seg_dir)
            return net_output

    def net_inference(self, train_dict, is_training):
        raise NotImplementedError

    def loss_func(self, net_outputs):
        real_logits = net_outputs[1]
        fake_logits = net_outputs[2]
        return self._loss_func(real_logits, fake_logits)

    def inference_loop(self, sess, coord, net_out):
        pass


    def stop(self):
        for sampler in self.get_sampler():
            sampler.close_all()

    def logs(self, train_dict, net_outputs):
        pass

    def training_ops(self, start_iter=0, end_iter=1):
        end_iter = max(start_iter, end_iter)
        for iter_i in range(start_iter, end_iter):
            yield iter_i, self._gradient_op

    def set_all_output_ops(self, output_op):
        pass

    def eval_variables(self):
        pass

    def process_output_values(self, values, is_training):
        # do nothing
        pass
        # if is_training:
        #    print(values)
        # else:
        #    print(values)

    def train_op_generator(self, apply_ops):
        pass

    def interpret_output(self, batch_output, is_training):
        if is_training:
            return True
        else:
            return self.output_decoder.decode_batch(
                batch_output['image'], batch_output['location'])
