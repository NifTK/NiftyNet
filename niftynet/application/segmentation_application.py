import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.image_windows_aggregator import GridSamplesAggregator
from niftynet.engine.sampler_grid import GridSampler
from niftynet.engine.sampler_uniform import UniformSampler
from niftynet.io.image_reader import ImageReader
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.loss import LossFunction
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.utilities import util_common as util

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
        elif name == "holistic_scalenet":
            from niftynet.network.holistic_scalenet import HolisticScaleNet
            return HolisticScaleNet
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
        if self.is_training:
            self.reader = ImageReader(SUPPORTED_INPUT)
        else:  # in the inference process use image input only
            self.reader = ImageReader(['image'])
        self.reader.initialise_reader(data_param, segmentation_param)

        if self.net_param.normalise_foreground_only:
            foreground_masking_layer = BinaryMaskingLayer(
                type=self.net_param.foreground_type,
                multimod_fusion=self.net_param.multimod_foreground_type,
                threshold=0.0)
        else:
            foreground_masking_layer = None

        mean_var_normaliser = MeanVarNormalisationLayer(
            field='image', binary_masking_func=foreground_masking_layer)
        if self.net_param.histogram_ref_file:
            histogram_normaliser = HistogramNormalisationLayer(
                field='image',
                modalities=vars(segmentation_param).get('image'),
                model_filename=self.net_param.histogram_ref_file,
                binary_masking_func=foreground_masking_layer,
                norm_type=self.net_param.norm_type,
                cutoff=self.net_param.cutoff,
                name='hist_norm_layer')
        else:
            histogram_normaliser = None

        if self.net_param.histogram_ref_file:
            label_normaliser = DiscreteLabelNormalisationLayer(
                field='label',
                modalities=vars(segmentation_param).get('label'),
                model_filename=self.net_param.histogram_ref_file)
        else:
            label_normaliser = None

        normalisation_layers = []
        if self.net_param.normalisation:
            normalisation_layers.append(histogram_normaliser)
        if self.net_param.whitening:
            normalisation_layers.append(mean_var_normaliser)
        if segmentation_param.label_normalisation:
            normalisation_layers.append(label_normaliser)

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

        volume_padding_layer = []
        if self.net_param.volume_padding_size:
            volume_padding_layer.append(PadLayer(
                field=SUPPORTED_INPUT,
                border=self.net_param.volume_padding_size))
        self.reader.add_preprocessing_layers(
            volume_padding_layer + normalisation_layers + augmentation_layers)

    def initialise_sampler(self, is_training):
        if is_training:
            self._sampler = UniformSampler(
                reader=self.reader,
                data_param=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=self.action_param.sample_per_volume)
        else:
            self._sampler = GridSampler(
                reader=self.reader,
                data_param=self.data_param,
                batch_size=self.net_param.batch_size,
                spatial_window_size=self.action_param.spatial_window_size,
                window_border=self.action_param.border)
        self._sampler = [self._sampler]

    def get_sampler(self):
        return self._sampler

    def initialise_network(self):
        num_classes = self.segmentation_param.num_classes
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

        self._net = NetFactory.create(self.net_param.name)(
            num_classes=num_classes,
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net_param.activation_function)

    def inference_sampler(self):
        pass
        # return samplseur

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 training_grads_collector=None):
        if self.is_training and training_grads_collector:
            device_id = training_grads_collector.current_tower_id
        else:
            device_id = 0
        data_dict = self.get_sampler()[0].pop_batch_op(device_id)
        net_out = self._net(data_dict['image'], self.is_training)

        if self.is_training:
            with tf.name_scope('Optimizer'):
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.action_param.lr)
            loss_func = LossFunction(
                n_class=self.segmentation_param.num_classes,
                loss_type=self.action_param.loss_type)
            data_loss = loss_func(
                pred=net_out,
                label=data_dict.get('label', None),
                weight_map=data_dict.get('weight', None))
            if self.net_param.decay > 0.0:
                reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                if reg_losses:
                    reg_loss = tf.reduce_mean(
                        [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                    loss = data_loss + reg_loss
            else:
                loss = data_loss
            grads = self.optimizer.compute_gradients(loss)
            # collecting gradients variables
            training_grads_collector.add_to_collection([grads])
            # collecting output variables
            outputs_collector.add_to_collection(
                var=data_loss, name='dice_loss',
                average_over_devices=True, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss, name='dice_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
        else:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
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
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=CONSOLE)
            self.output_decoder = GridSamplesAggregator(
                image_reader=self.reader,
                output_path=self.action_param.save_seg_dir,
                window_border=self.action_param.border,
                interp_order=self.action_param.output_interp_order)

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

    def inference_loop(self, sess, coord, net_out):
        if self._param.window_sampling in ['selective', 'uniform']:
            return self._inference_loop_patch(sess, coord, net_out)
        elif self._param.window_sampling in ['resize']:
            return self._inference_loop_resize(sess, coord, net_out)


    def training_ops(self, start_iter=0, end_iter=1):
        end_iter = max(start_iter, end_iter)
        for iter_i in range(start_iter, end_iter):
            yield iter_i, self._gradient_op

    def stop(self):
        for sampler in self.get_sampler():
            sampler.close_all()

    def interpret_output(self, batch_output, is_training):
        if not is_training:
            return self.output_decoder.decode_batch(
                batch_output['window'], batch_output['location'])
        else:
            return True
