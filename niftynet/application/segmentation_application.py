import time

import numpy as np
import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.grid_sampler import GridSampler
from niftynet.engine.resize_sampler import ResizeSampler
from niftynet.engine.selective_sampler import SelectiveSampler
from niftynet.engine.spatial_location_check import SpatialLocationCheckLayer
from niftynet.engine.uniform_sampler import UniformSampler
from niftynet.layer.binary_masking import BinaryMaskingLayer
# from niftynet.engine.volume_loader import VolumeLoaderLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.utilities.input_placeholders import ImagePatch

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

    def initialise_dataset_loader(self, data_param, segmentation_param):
        # read each line of csv files into an instance of Subject
        from niftynet.io.image_reader import ImageReader
        reader = ImageReader(SUPPORTED_INPUT)
        reader.initialise_reader(data_param, segmentation_param)

        foreground_masking_layer = BinaryMaskingLayer(
            type=self.net_param.mask_type,
            multimod_fusion=self.net_param.multimod_mask_type,
            threshold=0.0)
        histogram_normaliser = HistogramNormalisationLayer(
            field='image',
            modalities=segmentation_param['image'],
            models_filename=self.net_param.histogram_ref_file,
            binary_masking_func=foreground_masking_layer,
            norm_type=self.net_param.norm_type,
            cutoff=self.net_param.cutoff,
            name='hist_norm_layer')
        mean_var_normaliser = MeanVarNormalisationLayer(
            field='image',
            binary_masking_func=foreground_masking_layer)
        reader.add_preprocessing_layers([histogram_normaliser,
                                         mean_var_normaliser])
        from niftynet.engine.sampler_uniform import UniformSampler
        sampler = UniformSampler(reader,
                                 data_param,
                                 self.action_param.sample_per_volume)

    def initialise_sampler(self, is_training):
        pass

    def initialise_network(self, train_dict, is_training):
        pass

    def inference_sampler(self):
        self._inference_patch_holder = ImagePatch(
            spatial_rank=self._param.spatial_rank,
            image_size=self._param.image_size,
            label_size=self._param.label_size,
            weight_map_size=self._param.w_map_size,
            image_dtype=tf.float32,
            label_dtype=tf.int64,
            weight_map_dtype=tf.float32,
            num_image_modality=self._volume_loader.num_modality(0),
            num_label_modality=self._volume_loader.num_modality(1),
            num_weight_map=self._volume_loader.num_modality(2))

        # `patch` instance with image data only
        if self._param.window_sampling in ['uniform', 'selective']:
            sampling_grid_size = self._param.label_size - 2 * self._param.border
            assert sampling_grid_size > 0
            sampler = GridSampler(patch=self._inference_patch_holder,
                                  volume_loader=self._volume_loader,
                                  grid_size=sampling_grid_size,
                                  name='grid_sampler')
        elif self._param.window_sampling == 'resize':
            sampler = ResizeSampler(
                patch=self._inference_patch_holder,
                volume_loader=self._volume_loader,
                data_augmentation_methods=None,
                name="resize_sampler")
            # ops to resize image back
            self._ph = tf.placeholder(tf.float32, [None])
            self._sz = tf.placeholder(tf.int32, [None])
            reshaped = tf.image.resize_images(
                tf.reshape(self._ph, [1] + [self._param.label_size] * 2 + [-1]),
                self._sz[0:2])
            if self._param.spatial_rank == 3:
                reshaped = tf.reshape(reshaped, [1, self._sz[0] * self._sz[1],
                                                 self._param.label_size, -1])
                reshaped = tf.image.resize_images(reshaped,
                                                  [self._sz[0] * self._sz[1],
                                                   self._sz[2]])
            self._reshaped = tf.reshape(reshaped, self._sz)
        return sampler

    def sampler(self):
        patch_holder = ImagePatch(
            spatial_rank=self._param.spatial_rank,
            image_size=self._param.image_size,
            label_size=self._param.label_size,
            weight_map_size=self._param.w_map_size,
            image_dtype=tf.float32,
            label_dtype=tf.int64,
            weight_map_dtype=tf.float32,
            num_image_modality=self._volume_loader.num_modality(0),
            num_label_modality=self._volume_loader.num_modality(1),
            num_weight_map=self._volume_loader.num_modality(2))
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
