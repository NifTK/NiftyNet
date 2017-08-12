from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.graph_variables_collector import CONSOLE
from niftynet.engine.graph_variables_collector import TF_SUMMARIES
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

# from niftynet.engine.input_buffer import TrainEvalInputBuffer, DeployInputBuffer

SUPPORTED_INPUT = {'image'}


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
        self._loss_func = LossFunction(loss_type=self.action_param.loss_type)

        self.reader = None
        self.data_param = None
        self.gan_param = None

    def initialise_dataset_loader(self, data_param, gan_param):
        self.data_param = data_param
        self.gan_param = gan_param

        # read each line of csv files into an instance of Subject
        if self.is_training:
            self.reader = ImageReader(SUPPORTED_INPUT)
        else:  # in the inference process use image input only
            self.reader = ImageReader(['image'])
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
            rand_flip_layer = RandomFlipLayer(
                flip_axes=self.action_param.flip_axes)
            rand_scaling_layer = RandomSpatialScalingLayer(
                min_percentage=self.action_param.scaling_percentage[0],
                max_percentage=self.action_param.scaling_percentage[1])
            rand_rotate_layer = RandomRotationLayer(
                min_angle=self.action_param.rotation_angle[0],
                max_angle=self.action_param.rotation_angle[1])

            if self.action_param.random_flip:
                augmentation_layers.append(rand_flip_layer)
            if self.action_param.spatial_scaling:
                augmentation_layers.append(rand_scaling_layer)
            if self.action_param.rotation:
                augmentation_layers.append(rand_rotate_layer)

        self.reader.add_preprocessing_layers(
            normalisation_layers + augmentation_layers)

    def inference_sampler(self):
        pass
        # assert self._volume_loader is not None, \
        #     "Please call initialise_dataset_loader first"
        # self._inference_patch_holder = GANPatch(
        #     spatial_rank=self._param.spatial_rank,
        #     image_size=self._param.image_size,
        #     noise_size=self._param.noise_size,
        #     conditioning_size=self._param.conditioning_size,
        #     num_image_modality=self._volume_loader.num_modality(0))
        #
        # sampler = GANSampler(
        #     patch=self._inference_patch_holder,
        #     volume_loader=self._volume_loader,
        #     data_augmentation_methods=None)
        # # ops to resize image back
        # self._ph = tf.placeholder(tf.float32, [None])
        # self._sz = tf.placeholder(tf.int32, [None])
        # reshaped = tf.image.resize_images(
        #     tf.reshape(self._ph, [1] + [self._param.image_size] * 2 + [-1]),
        #     self._sz[0:2])
        # if self._param.spatial_rank == 3:
        #     reshaped = tf.reshape(reshaped, [1, self._sz[0] * self._sz[1],
        #                                      self._param.image_size, -1])
        #     reshaped = tf.image.resize_images(reshaped,
        #                                       [self._sz[0] * self._sz[1],
        #                                        self._sz[2]])
        # self._reshaped = tf.reshape(reshaped, self._sz)
        # input_buffer= DeployInputBuffer(
        #     batch_size=self._param.batch_size,
        #     capacity=max(self._param.queue_length, self._param.batch_size),
        #     sampler=sampler)
        # return input_buffer

    def get_sampler(self):
        return self._sampler

    def initialise_sampler(self, is_training):
        if is_training:
            if self.gan_param.window_sampling == "resize":
                self._sampler = ResizeSampler(
                    reader=self.reader,
                    data_param=self.data_param,
                    batch_size=self.net_param.batch_size,
                    windows_per_image=self.action_param.sample_per_volume)
        else:
            self._sampler = self.inference_sampler()
        self._sampler = [self._sampler]

    def training_sampler(self):
        pass
        # assert self._volume_loader is not None, \
        #     "Please call initialise_dataset_loader first"
        # patch_holder = GANPatch(
        #     spatial_rank=self._param.spatial_rank,
        #     image_size=self._param.image_size,
        #     noise_size=self._param.noise_size,
        #     conditioning_size=self._param.conditioning_size,
        #     num_image_modality=self._volume_loader.num_modality(0),
        #     num_conditioning_modality=self._volume_loader.num_modality(1))
        #
        # # defines how to generate samples of the training element from volume
        # with tf.name_scope('Sampling'):
        #     sampler = GANSampler(
        #         patch=patch_holder,
        #         volume_loader=self._volume_loader,
        #         data_augmentation_methods=None)
        # input_buffer = TrainEvalInputBuffer(
        #     batch_size=self._param.batch_size,
        #     capacity=max(self._param.queue_length, self._param.batch_size),
        #     sampler=sampler,
        #     shuffle=True)
        # return input_buffer

    def initialise_network(self):
        self._net = GanFactory.create(self.net_param.name)()

    def connect_data_and_network(
            self, outputs_collector=None, training_grads_collector=None):

        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.action_param.lr)
        if self.is_training:
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
            conditioning = None
            net_output = self._net(noise,
                                   images,
                                   conditioning,
                                   self.is_training)

            lossG, lossD = self.loss_func(data_dict, net_output)
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
            raise NotImplementedError
            # data_dict = self._sampler.pop_batch_op()
            # images = data_dict['images']
            # net_output = self._net(images, False)
            # return net_output

    def net_inference(self, train_dict, is_training):
        raise NotImplementedError
        # if not self._net:
        #    self._net = self._net_class()
        # net_outputs = self._net(train_dict['images'], is_training)
        # return self._post_process_outputs(net_outputs), train_dict['info']

    def loss_func(self, train_dict, net_outputs):
        real_logits = net_outputs[1]
        fake_logits = net_outputs[2]
        lossG = self._loss_func(fake_logits, True)
        lossD = self._loss_func(real_logits, True) + self._loss_func(
            fake_logits, False)
        return lossG, lossD

    # def train(self, train_dict):
    #    """
    #    Returns a list of possible compute_gradients ops to be run each training iteration.
    #    Default implementation returns gradients for all variables from one Adam optimizer
    #    """
    #    with tf.name_scope('Optimizer'):
    #        self.optimizer = tf.train.AdamOptimizer(
    #            learning_rate=self._param.lr, )
    #    net_outputs = self.net(train_dict, is_training=True)
    #    with tf.name_scope('Loss'):
    #        lossG, lossD = self.loss_func(train_dict, net_outputs)
    #        if self._param.decay > 0:
    #            reg_losses = tf.get_collection(
    #                tf.GraphKeys.REGULARIZATION_LOSSES)
    #            if reg_losses:
    #                reg_loss = tf.reduce_mean([tf.reduce_mean(reg_loss)
    #                                           for reg_loss in reg_losses])
    #                lossD = lossD + reg_loss
    #                lossG = lossG + reg_loss
    #    # Averages are in name_scope for Tensorboard naming; summaries are outside for console naming
    #    logs = [['lossD', lossD], ['lossG', lossG]]
    #    with tf.name_scope('ConsoleLogging'):
    #        logs += self.logs(train_dict, net_outputs)
    #    for tag, val in logs:
    #        tf.summary.scalar(tag, val, [logging.CONSOLE, logging.LOG])
    #    with tf.name_scope('ComputeGradients'):
    #        generator_variables = tf.get_collection(
    #            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    #        discriminator_variables = tf.get_collection(
    #            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    #        grads = [self.optimizer.compute_gradients(lossG,
    #                                                  var_list=generator_variables),
    #                 self.optimizer.compute_gradients(lossD,
    #                                                  var_list=discriminator_variables)]
    #    # add compute gradients ops for each type of optimizer_op
    #    return grads

    def inference_loop(self, sess, coord, net_out):
        pass
        # all_saved_flag = False
        # img_id, pred_img, subject_i = None, None, None
        # spatial_rank = self._inference_patch_holder.spatial_rank
        # while True:
        #    local_time = time.time()
        #    if coord.should_stop():
        #        break
        #    seg_maps, spatial_info = sess.run(net_out)
        #    # go through each one in a batch
        #    for batch_id in range(seg_maps.shape[0]):
        #        img_id = spatial_info[batch_id, 0]
        #        subject_i = self._volume_loader.get_subject(img_id)
        #        pred_img = subject_i.matrix_like_input_data_5d(
        #            spatial_rank=spatial_rank,
        #            n_channels=self._num_output_channels_func(),
        #            interp_order=self._param.output_interp_order)
        #        predictions = seg_maps[batch_id]
        #        while predictions.ndim < pred_img.ndim:
        #            predictions = np.expand_dims(predictions, axis=-1)

        #        # assign predicted patch to the allocated output volume
        #        origin = spatial_info[
        #                 batch_id, 1:(1 + int(np.floor(spatial_rank)))]

        #        i_spatial_rank = int(np.ceil(spatial_rank))
        #        output_size = [self._param.image_size] * i_spatial_rank + [1]
        #        pred_size = pred_img.shape[0:i_spatial_rank] + [1]
        #        zoom = [d / p for p, d in zip(output_size, pred_size)]
        #        ph = np.reshape(predictions, [-1])
        #        pred_img = sess.run(
        #                [self._reshaped],
        #                feed_dict={self._ph: ph, self._sz: pred_img.shape})[0]
        #        subject_i.save_network_output(
        #            pred_img,
        #            self._param.save_seg_dir,
        #            self._param.output_interp_order)

        #        if self._inference_patch_holder.is_stopping_signal(
        #                spatial_info[batch_id]):
        #            print('received finishing batch')
        #            all_saved_flag = True
        #            return all_saved_flag

        #            # try to expand prediction dims to match the output volume
        #    print('processed {} image patches ({:.3f}s)'.format(
        #        len(spatial_info), time.time() - local_time))
        # return all_saved_flag

    def stop(self):
        for sampler in self.get_sampler():
            sampler.close_all()

    def logs(self, train_dict, net_outputs):
        pass
        # return []

    def training_ops(self, start_iter=0, end_iter=1):
        end_iter = max(start_iter, end_iter)
        for iter_i in range(start_iter, end_iter):
            yield iter_i, self._gradient_op

    def set_all_output_ops(self, output_op):
        pass
        # self._output_op = output_op

    def eval_variables(self):
        pass
        # return self._output_op[0][0][1]

    def process_output_values(self, values, is_training):
        # do nothing
        pass
        # if is_training:
        #    print(values)
        # else:
        #    print(values)

    def train_op_generator(self, apply_ops):
        pass
        # while True:
        #    yield apply_ops
