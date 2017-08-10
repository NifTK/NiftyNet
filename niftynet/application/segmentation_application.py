import time

import numpy as np
import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.grid_sampler import GridSampler
from niftynet.engine.resize_sampler import ResizeSampler
from niftynet.engine.selective_sampler import SelectiveSampler
from niftynet.engine.spatial_location_check import SpatialLocationCheckLayer
from niftynet.engine.sampler_uniform import UniformSampler
from niftynet.layer.loss import LossFunction

from niftynet.io.image_reader import ImageReader
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.post_processing import PostProcessingLayer

from niftynet.io.misc_io import remove_time_dim
from niftynet.utilities import misc_common as util

SUPPORTED_INPUT = {'image', 'label', 'weight'}


class NetFactory(object):
    @staticmethod
    def create(name):
        if name == "highres3dnet":
            from niftynet.network.highres3dnet import HighRes3DNet
            return HighRes3DNet
        if name == "highres3dnet_small":
            from niftynet.network.highres3dnet_small import HighRes3DNetSmall
            return HighRes3DNetSmall
        if name == "highres3dnet_large":
            from niftynet.network.highres3dnet_large import HighRes3DNetLarge
            return HighRes3DNetLarge
        elif name == "toynet":
            from niftynet.network.toynet import ToyNet
            return ToyNet
        elif name == "unet":
            from niftynet.network.unet import UNet3D
            return UNet3D
        elif name == "vnet":
            from niftynet.network.vnet import VNet
            return VNet
        elif name == "dense_vnet":
            from niftynet.network.dense_vnet import DenseVNet
            return DenseVNet
        elif name == "deepmedic":
            from niftynet.network.deepmedic import DeepMedic
            return DeepMedic
        elif name == "scalenet":
            from niftynet.network.scalenet import ScaleNet
            return ScaleNet
        else:
            print("network: \"{}\" not implemented".format(name))
            raise NotImplementedError


class SegmentationApplication(BaseApplication):
    # def __init__(self, net_class, param, volume_loader):
    #     self._net_class = net_class
    #     self._param = param
    #     self._volume_loader = volume_loader
    #     self._loss_func = LossFunction(n_class=self._param.num_classes,
    #                                    loss_type=self._param.loss_type)
    #     self.num_objectives = 1
    #     w_regularizer, b_regularizer = self.regularizers()
    #     self._net = net_class(num_classes=self._param.num_classes,
    #                           w_regularizer=w_regularizer,
    #                           b_regularizer=b_regularizer,
    #                           acti_func=self._param.activation_function)
    #     self._net_inference = net_class(num_classes=self._param.num_classes,
    #                                     acti_func=self._param.activation_function)

    def set_model_param(self, net_param, action_param, is_training):
        self.is_training = is_training
        self.net_param = net_param
        self.action_param = action_param
        self.reader = None
        self.data_param = None
        self.segmentation_param = None

    def initialise_dataset_loader(self, data_param, segmentation_param):
        self.data_param = data_param
        self.segmentation_param = segmentation_param
        # read each line of csv files into an instance of Subject
        self.reader = ImageReader(SUPPORTED_INPUT)
        self.reader.initialise_reader(data_param, segmentation_param)

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
                modalities=vars(segmentation_param).get('image'),
                model_filename=self.net_param.histogram_ref_file,
                binary_masking_func=foreground_masking_layer,
                norm_type=self.net_param.norm_type,
                cutoff=self.net_param.cutoff,
                name='hist_norm_layer')
            label_normaliser = DiscreteLabelNormalisationLayer(
                field='label',
                modalities=vars(segmentation_param).get('label'),
                model_filename=self.net_param.histogram_ref_file)
        else:
            histogram_normaliser = None
            label_normaliser = None

        normalisation_layers = []
        if self.net_param.normalisation:
            normalisation_layers.append(histogram_normaliser)
        if self.net_param.whitening:
            normalisation_layers.append(mean_var_normaliser)
        if segmentation_param.label_normalisation:
            normalisation_layers.append(label_normaliser)

        rand_flip_layer = RandomFlipLayer(
            flip_axes=self.action_param.flip_axes)
        rand_scaling_layer = RandomSpatialScalingLayer(
            min_percentage=self.action_param.scaling_percentage[0],
            max_percentage=self.action_param.scaling_percentage[1])
        rand_rotate_layer = RandomRotationLayer(
            min_angle=self.action_param.rotation_angle[0],
            max_angle=self.action_param.rotation_angle[1])

        augmentation_layers = []
        if self.is_training and self.action_param.random_flip:
            augmentation_layers.append(rand_flip_layer)
        if self.is_training and self.action_param.spatial_scaling:
            augmentation_layers.append(rand_scaling_layer)
        if self.is_training and self.action_param.rotation:
            augmentation_layers.append(rand_rotate_layer)

        self.reader.add_preprocessing_layers(
               normalisation_layers+augmentation_layers)


    def initialise_sampler(self, is_training):
        self._sampler = UniformSampler(self.reader,
                                       self.data_param,
                                       self.action_param.sample_per_volume)
    def get_sampler(self):
        return self._sampler

    def initialise_network(self):
        num_classes = self.segmentation_param.num_classes
        # TODO regularisation
        self._net = NetFactory.create(self.net_param.name)(
            num_classes=num_classes)

    def inference_sampler(self):
        return sampler

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 training_grads_collector=None):
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.action_param.lr)
        device_id = training_grads_collector.current_tower_id
        data_dict = self._sampler.pop_batch_op(device_id,
                                               self.net_param.batch_size)
        for field in data_dict:
            data_dict[field] = remove_time_dim(data_dict[field])
        net_out = self._net(data_dict['image'], self.is_training)
        loss_func = LossFunction(n_class=self.segmentation_param.num_classes,
                                 loss_type=self.action_param.loss_type)
        loss = loss_func(pred=net_out,
                         label=data_dict.get('label', None),
                         weight_map=data_dict.get('weight', None))

        grads = self.optimizer.compute_gradients(loss)
        training_grads_collector.add_to_collection([grads])
        outputs_collector.print_to_console(var=loss,
                                           name='dice_loss',
                                           average_over_devices=True)
        outputs_collector.print_to_tf_summary(var=loss,
                                              name='dice_loss',
                                              average_over_devices=True,
                                              summary_type='scalar')
        return net_out

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

    def sampler(self):
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
            if self._param.window_sampling == 'uniform':
                sampler = UniformSampler(patch=patch_holder,
                                         volume_loader=self._volume_loader,
                                         patch_per_volume=self._param.sample_per_volume,
                                         data_augmentation_methods=augmentations,
                                         name='uniform_sampler')
            elif self._param.window_sampling == 'selective':
                # TODO check self._param, this is for segmentation problems only
                spatial_location_check = SpatialLocationCheckLayer(
                    compulsory=((0), (0)),
                    minimum_ratio=self._param.min_sampling_ratio,
                    min_numb_labels=self._param.min_numb_labels,
                    padding=self._param.border,
                    name='spatial_location_check')
                sampler = SelectiveSampler(
                    patch=patch_holder,
                    volume_loader=self._volume_loader,
                    spatial_location_check=spatial_location_check,
                    data_augmentation_methods=None,
                    patch_per_volume=self._param.sample_per_volume,
                    name="selective_sampler")
            elif self._param.window_sampling == 'resize':
                sampler = ResizeSampler(
                    patch=patch_holder,
                    volume_loader=self._volume_loader,
                    data_augmentation_methods=None,
                    name="resize_sampler")
        return sampler

    def net(self, train_dict, is_training):
        return self._net(train_dict['Sampling/images'], is_training)

    def net_inference(self, train_dict, is_training):
        net_outputs = self._net(train_dict['images'], is_training)
        return self._post_process_outputs(net_outputs), train_dict['info']

    def loss_func(self, train_dict, net_outputs):
        if "weight_maps" in train_dict:
            weight_maps = train_dict['Sampling/weight_maps']
        else:
            weight_maps = None
        return self._loss_func(net_outputs, train_dict['Sampling/labels'],
                               weight_maps)

    def _post_process_outputs(self, net_outputs):
        # converting logits into final output for
        # classification probabilities or argmax classification labels
        if self._param.output_prob and self._param.num_classes > 1:
            post_process_layer = PostProcessingLayer(
                'SOFTMAX', num_classes=self._param.num_classes)
        elif not self._param.output_prob and self._param.num_classes > 1:
            post_process_layer = PostProcessingLayer(
                'ARGMAX', num_classes=self._param.num_classes)
        else:
            post_process_layer = PostProcessingLayer(
                'IDENTITY', num_classes=self._param.num_classes)
        self._num_output_channels_func = post_process_layer.num_output_channels
        return post_process_layer(net_outputs)

    def inference_loop(self, sess, coord, net_out):
        if self._param.window_sampling in ['selective', 'uniform']:
            return self._inference_loop_patch(sess, coord, net_out)
        elif self._param.window_sampling in ['resize']:
            return self._inference_loop_resize(sess, coord, net_out)

    def _inference_loop_resize(self, sess, coord, net_out):
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
                zoom = [d / p for p, d in
                        zip([self._param.label_size] * i_spatial_rank,
                            pred_img.shape[0:i_spatial_rank])] + [1, 1]
                # tic=time.time()
                # pred_img[...] = scipy.ndimage.interpolation.zoom(predictions, zoom)
                # print(time.time()-tic)
                tic = time.time()
                pred_img = sess.run([self._reshaped], feed_dict={
                    self._ph: np.reshape(predictions, [-1]),
                    self._sz: pred_img.shape})[0]
                print(time.time() - tic)
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

    def _inference_loop_patch(self, sess, coord, ):
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
                if spatial_info[batch_id, 0] != img_id:
                    # when subject_id changed
                    # save current map and reset cumulative map variable
                    if subject_i is not None:
                        subject_i.save_network_output(
                            pred_img,
                            self._param.save_seg_dir,
                            self._param.output_interp_order)

                    if self._inference_patch_holder.is_stopping_signal(
                            spatial_info[batch_id]):
                        print('received finishing batch')
                        all_saved_flag = True
                        return all_saved_flag

                    img_id = spatial_info[batch_id, 0]
                    subject_i = self._volume_loader.get_subject(img_id)
                    pred_img = subject_i.matrix_like_input_data_5d(
                        spatial_rank=spatial_rank,
                        n_channels=self._num_output_channels,
                        interp_order=self._param.output_interp_order)

                # try to expand prediction dims to match the output volume
                predictions = seg_maps[batch_id]
                while predictions.ndim < pred_img.ndim:
                    predictions = np.expand_dims(predictions, axis=-1)

                # assign predicted patch to the allocated output volume
                origin = spatial_info[
                         batch_id, 1:(1 + int(np.floor(spatial_rank)))]

                # indexing within the patch
                assert self._param.label_size >= self._param.border * 2
                p_ = self._param.border
                _p = self._param.label_size - self._param.border

                # indexing relative to the sampled volume
                assert self._param.image_size >= self._param.label_size
                image_label_size_diff = self._param.image_size - self._param.label_size
                s_ = self._param.border + int(image_label_size_diff / 2)
                _s = s_ + self._param.label_size - 2 * self._param.border
                # absolute indexing in the prediction volume
                dest_start, dest_end = (origin + s_), (origin + _s)

                assert np.all(dest_start >= 0)
                img_dims = pred_img.shape[0:int(np.floor(spatial_rank))]
                assert np.all(dest_end <= img_dims)
                if spatial_rank == 3:
                    x_, y_, z_ = dest_start
                    _x, _y, _z = dest_end
                    pred_img[x_:_x, y_:_y, z_:_z, ...] = \
                        predictions[p_:_p, p_:_p, p_:_p, ...]
                elif spatial_rank == 2:
                    x_, y_ = dest_start
                    _x, _y = dest_end
                    pred_img[x_:_x, y_:_y, ...] = \
                        predictions[p_:_p, p_:_p, ...]
                elif spatial_rank == 2.5:
                    x_, y_ = dest_start
                    _x, _y = dest_end
                    z_ = spatial_info[batch_id, 3]
                    pred_img[x_:_x, y_:_y, z_:(z_ + 1), ...] = \
                        predictions[p_:_p, p_:_p, ...]
                else:
                    raise ValueError("unsupported spatial rank")
            print('processed {} image patches ({:.3f}s)'.format(
                len(spatial_info), time.time() - local_time))
        return all_saved_flag

    def logs(self, train_dict, net_outputs):
        predictions = net_outputs
        labels = train_dict['Sampling/labels']
        return [['miss', tf.reduce_mean(tf.cast(
            tf.not_equal(tf.argmax(predictions, -1), labels[..., 0]),
            dtype=tf.float32))]]

    def training_ops(self, start_iter=0, end_iter=1):
        end_iter = max(start_iter, end_iter)
        for iter_i in range(start_iter, end_iter):
            yield iter_i, self._gradient_op
    def stop(self):
        self._sampler.close_all()
