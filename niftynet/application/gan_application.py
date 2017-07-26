from __future__ import absolute_import, print_function
from niftynet.engine.gan_sampler import GANSampler
import time

import numpy as np
import tensorflow as tf

import niftynet.utilities.param_shortcuts_expanding as param_util
from niftynet.application.base_application import BaseApplication
from niftynet.engine import network_tensor_collector as logging
from niftynet.engine.gan_sampler import GANSampler
from niftynet.engine.volume_loader import VolumeLoaderLayer
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.gan_loss import LossFunction
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.utilities.csv_table import CSVTable
from niftynet.utilities.input_placeholders import GANPatch
from niftynet.engine.input_buffer import TrainEvalInputBuffer, DeployInputBuffer
from niftynet.utilities import misc_common as util


class GanNetFactory(object):
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

    def set_param(self, param):
        self._param = param
        self._volume_loader = None
        self._sampler = None
        self._loss_func = LossFunction(loss_type=self._param.loss_type)

    # def __init__(self, param):
    #     # super(GANApplication, self).__init__()
    #     self._net_class = None
    #     self._param = param
    #     self._volume_loader = None
    #     self._loss_func = LossFunction(loss_type=self._param.loss_type)
    #     self.num_objectives = 2
    #     self._net = None
    #     self._net_class_module = NetFactory.create(param.net_name)

    def initialise_dataset_loader(self, csv_dict):
        # read each line of csv files into an instance of Subject
        csv_loader = CSVTable(csv_dict=csv_dict, allow_missing=True)

        # expanding user input parameters
        spatial_padding = param_util.expand_padding_params(
            self._param.volume_padding_size, self._param.spatial_rank)
        interp_order = (self._param.image_interp_order,
                        self._param.conditioning_interp_order)

        # define layers of volume-level normalisation
        normalisation_layers = []
        if self._param.normalisation:
            hist_norm = HistogramNormalisationLayer(
                models_filename=self._param.histogram_ref_file,
                binary_masking_func=BinaryMaskingLayer(
                    type=self._param.mask_type,
                    multimod_fusion=self._param.multimod_mask_type),
                norm_type=self._param.norm_type,
                cutoff=(self._param.cutoff_min, self._param.cutoff_max))
            normalisation_layers.append(hist_norm)
        if self._param.whitening:
            mean_std_norm = MeanVarNormalisationLayer(
                binary_masking_func=BinaryMaskingLayer(
                    type=self._param.mask_type,
                    multimod_fusion=self._param.multimod_mask_type))
            normalisation_layers.append(mean_std_norm)

        # define how to load image volumes
        self._volume_loader = VolumeLoaderLayer(
            csv_loader,
            standardisor=normalisation_layers,
            is_training=(self._param.action == "train"),
            do_reorientation=self._param.reorientation,
            do_resampling=self._param.resampling,
            spatial_padding=spatial_padding,
            interp_order=interp_order)

    def inference_sampler(self):
        assert self._volume_loader is not None, \
            "Please call initialise_dataset_loader first"
        self._inference_patch_holder = GANPatch(
            spatial_rank=self._param.spatial_rank,
            image_size=self._param.image_size,
            noise_size=self._param.noise_size,
            conditioning_size=self._param.conditioning_size,
            num_image_modality=self._volume_loader.num_modality(0))

        sampler = GANSampler(
            patch=self._inference_patch_holder,
            volume_loader=self._volume_loader,
            data_augmentation_methods=None)
        # ops to resize image back
        self._ph = tf.placeholder(tf.float32, [None])
        self._sz = tf.placeholder(tf.int32, [None])
        reshaped = tf.image.resize_images(
            tf.reshape(self._ph, [1] + [self._param.image_size] * 2 + [-1]),
            self._sz[0:2])
        if self._param.spatial_rank == 3:
            reshaped = tf.reshape(reshaped, [1, self._sz[0] * self._sz[1],
                                             self._param.image_size, -1])
            reshaped = tf.image.resize_images(reshaped,
                                              [self._sz[0] * self._sz[1],
                                               self._sz[2]])
        self._reshaped = tf.reshape(reshaped, self._sz)
        input_buffer= DeployInputBuffer(
            batch_size=self._param.batch_size,
            capacity=max(self._param.queue_length, self._param.batch_size),
            sampler=sampler)
        return input_buffer

    def get_sampler(self):
        return self._sampler

    def initialise_sampler(self, is_training):
        if is_training:
            self._sampler = self.training_sampler()
        else:
            self._sampler = self.inference_sampler()

    def training_sampler(self):
        assert self._volume_loader is not None, \
            "Please call initialise_dataset_loader first"
        patch_holder = GANPatch(
            spatial_rank=self._param.spatial_rank,
            image_size=self._param.image_size,
            noise_size=self._param.noise_size,
            conditioning_size=self._param.conditioning_size,
            num_image_modality=self._volume_loader.num_modality(0),
            num_conditioning_modality=self._volume_loader.num_modality(1))
        # defines data augmentation for training
        augmentations = []
        if self._param.rotation:
            from niftynet.layer.rand_rotation import RandomRotationLayer
            augmentations.append(RandomRotationLayer(
                min_angle=self._param.min_angle,
                max_angle=self._param.max_angle))
        if self._param.spatial_scaling:
            from niftynet.layer.rand_spatial_scaling import \
                RandomSpatialScalingLayer
            augmentations.append(RandomSpatialScalingLayer(
                min_percentage=self._param.min_percentage,
                max_percentage=self._param.max_percentage))
        # defines how to generate samples of the training element from volume
        with tf.name_scope('Sampling'):
            sampler = GANSampler(
                patch=patch_holder,
                volume_loader=self._volume_loader,
                data_augmentation_methods=None)
        input_buffer = TrainEvalInputBuffer(
            batch_size=self._param.batch_size,
            capacity=max(self._param.queue_length, self._param.batch_size),
            sampler=sampler,
            shuffle=True)
        return input_buffer

    def initialise_network(self):
        self._net = GanNetFactory.create(self._param.net_name)()

    def set_network_update_op(self, gradients):
        grad_list_depth = util.list_depth_count(gradients)
        if grad_list_depth == 3:
            # nested depth 3 means: gradients list is nested in terms of:
            # list of networks -> list of network variables
            self._gradient_op = [self.optimizer.apply_gradients(grad)
                                 for grad in gradients]
        elif grad_list_depth == 2:
            # nested depth 2 means:
            # gradients list is a list of variables
            self._gradient_op = self.optimizer.apply_gradients(gradients)
        else:
            raise NotImplementedError(
                'This app supports updating a network, or list of networks')

    def connect_data_and_network(
        self, outputs_collector=None, training_grads_collector=None):

        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self._param.lr)
        is_training = training_grads_collector is not None

        if is_training:
            # a new pop_batch_op for each gpu tower
            device_id = training_grads_collector.current_tower_id
            data_dict = self._sampler.pop_batch_op(device_id)

            noise = data_dict['Sampling/noise']
            images = data_dict['Sampling/images']
            conditioning = data_dict.get('Sampling/conditioning', None)
            net_output = self._net(noise, images, conditioning, is_training)

            lossG, lossD = self.loss_func(data_dict, net_output)
            if self._param.decay > 0:
                reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                if reg_losses is not None and len(reg_losses) > 0:
                    reg_loss = tf.reduce_mean(
                        [tf.reduce_mean(l_reg) for l_reg in reg_losses])
                    lossD = lossD + reg_loss
                    lossG = lossG + reg_loss

            # variables to display in STDOUT
            outputs_collector.print_to_console(
                var=lossD, name='lossD', average_over_devices=True)
            outputs_collector.print_to_console(
                var=lossG, name='lossG', average_over_devices=False)
            # variables to display in tensorboard
            outputs_collector.print_to_tf_summary(
                var=lossG, name='lossG',
                average_over_devices=False, summary_type='scalar')
            outputs_collector.print_to_tf_summary(
                var=lossG, name='lossD',
                average_over_devices=True, summary_type='scalar')

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
            #data_dict = self._sampler.pop_batch_op()
            #images = data_dict['images']
            #net_output = self._net(images, False)
            #return net_output

    def net_inference(self, train_dict, is_training):
        raise NotImplementedError
        #if not self._net:
        #    self._net = self._net_class()
        #net_outputs = self._net(train_dict['images'], is_training)
        #return self._post_process_outputs(net_outputs), train_dict['info']

    def loss_func(self, train_dict, net_outputs):
        real_logits = net_outputs[1]
        fake_logits = net_outputs[2]
        lossG = self._loss_func(fake_logits, True)
        lossD = self._loss_func(real_logits, True) + self._loss_func(
            fake_logits, False)
        return lossG, lossD

    def train(self, train_dict):
        """
        Returns a list of possible compute_gradients ops to be run each training iteration.
        Default implementation returns gradients for all variables from one Adam optimizer
        """
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self._param.lr, )
        net_outputs = self.net(train_dict, is_training=True)
        with tf.name_scope('Loss'):
            lossG, lossD = self.loss_func(train_dict, net_outputs)
            if self._param.decay > 0:
                reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                if reg_losses:
                    reg_loss = tf.reduce_mean([tf.reduce_mean(reg_loss)
                                               for reg_loss in reg_losses])
                    lossD = lossD + reg_loss
                    lossG = lossG + reg_loss
        # Averages are in name_scope for Tensorboard naming; summaries are outside for console naming
        logs = [['lossD', lossD], ['lossG', lossG]]
        with tf.name_scope('ConsoleLogging'):
            logs += self.logs(train_dict, net_outputs)
        for tag, val in logs:
            tf.summary.scalar(tag, val, [logging.CONSOLE, logging.LOG])
        with tf.name_scope('ComputeGradients'):
            generator_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            discriminator_variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
            grads = [self.optimizer.compute_gradients(lossG,
                                                      var_list=generator_variables),
                     self.optimizer.compute_gradients(lossD,
                                                      var_list=discriminator_variables)]
        # add compute gradients ops for each type of optimizer_op
        return grads

    def inference_loop(self, sess, coord, net_out):
        all_saved_flag = False
        img_id, pred_img, subject_i = None, None, None
        spatial_rank = self._inference_patch_holder.spatial_rank
        while True:
            local_time = time.time()
            if coord.should_stop():
                break
            seg_maps, spatial_info = sess.run(net_out)
            # go through each one in a batch
            for batch_id in range(seg_maps.shape[0]):
                img_id = spatial_info[batch_id, 0]
                subject_i = self._volume_loader.get_subject(img_id)
                pred_img = subject_i.matrix_like_input_data_5d(
                    spatial_rank=spatial_rank,
                    n_channels=self._num_output_channels_func(),
                    interp_order=self._param.output_interp_order)
                predictions = seg_maps[batch_id]
                while predictions.ndim < pred_img.ndim:
                    predictions = np.expand_dims(predictions, axis=-1)

                # assign predicted patch to the allocated output volume
                origin = spatial_info[
                         batch_id, 1:(1 + int(np.floor(spatial_rank)))]

                i_spatial_rank = int(np.ceil(spatial_rank))
                output_size = [self._param.image_size] * i_spatial_rank + [1]
                pred_size = pred_img.shape[0:i_spatial_rank] + [1]
                zoom = [d / p for p, d in zip(output_size, pred_size)]
                ph = np.reshape(predictions, [-1])
                pred_img = sess.run([self._reshaped], feed_dict={self._ph: ph,
                                                                 self._sz: pred_img.shape})[
                    0]
                subject_i.save_network_output(
                    pred_img,
                    self._param.save_seg_dir,
                    self._param.output_interp_order)

                if self._inference_patch_holder.is_stopping_signal(
                        spatial_info[batch_id]):
                    print('received finishing batch')
                    all_saved_flag = True
                    return all_saved_flag

                    # try to expand prediction dims to match the output volume
            print('processed {} image patches ({:.3f}s)'.format(
                len(spatial_info), time.time() - local_time))
        return all_saved_flag

    def stop(self):
        self._sampler.close_all()

    def logs(self, train_dict, net_outputs):
        return []

    def training_ops(self, start_iter=0, end_iter=1):
        for iter_i in range(start_iter, end_iter):
            yield iter_i, self._gradient_op

    def set_all_output_ops(self, output_op):
        self._output_op = output_op

    def eval_variables(self):
        return self._output_op[0][0][1]

    def process_output_values(self, values, is_training):
        # do nothing
        pass
        #if is_training:
        #    print(values)
        #else:
        #    print(values)

    def train_op_generator(self, apply_ops):
        while True:
            yield apply_ops
