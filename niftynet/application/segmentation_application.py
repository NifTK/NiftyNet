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
            tf.logging.fatal("network: \"{}\" not implemented".format(name))
            raise NotImplementedError


class SegmentationApplication(BaseApplication):
    def __init__(self):
        BaseApplication.__init__(self)
        tf.logging.info('starting segmentation application')
        self.is_training = True

        self.net_param = None
        self.action_param = None

        self.data_param = None
        self.segmentation_param = None

    def set_app_param(self, net_param, action_param, is_training):
        self.is_training = is_training
        self.net_param = net_param
        self.action_param = action_param

    def initialise_dataset_loader(self, data_param=None, task_param=None):
        self.data_param = data_param
        self.segmentation_param = task_param

        # read each line of csv files into an instance of Subject
        if self.is_training:
            self.reader = ImageReader(SUPPORTED_INPUT)
        else:  # in the inference process use image input only
            self.reader = ImageReader(['image'])
        self.reader.initialise_reader(data_param, task_param)

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
                modalities=vars(task_param).get('image'),
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
                modalities=vars(task_param).get('label'),
                model_filename=self.net_param.histogram_ref_file)
        else:
            label_normaliser = None

        normalisation_layers = []
        if self.net_param.normalisation:
            normalisation_layers.append(histogram_normaliser)
        if self.net_param.whitening:
            normalisation_layers.append(mean_var_normaliser)
        if task_param.label_normalisation:
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

    def initialise_sampler(self):
        if self.is_training:
            self.sampler = UniformSampler(
                reader=self.reader,
                data_param=self.data_param,
                batch_size=self.net_param.batch_size,
                windows_per_image=self.action_param.sample_per_volume)
        else:
            self.sampler = GridSampler(
                reader=self.reader,
                data_param=self.data_param,
                batch_size=self.net_param.batch_size,
                spatial_window_size=self.action_param.spatial_window_size,
                window_border=self.action_param.border)
        self.sampler = [self.sampler]

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

        self.net = NetFactory.create(self.net_param.name)(
            num_classes=num_classes,
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net_param.activation_function)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        if self.is_training and gradients_collector:
            device_id = gradients_collector.current_tower_id
        else:
            device_id = 0
        data_dict = self.get_sampler()[0].pop_batch_op(device_id)
        image = tf.cast(data_dict['image'], tf.float32)
        net_out = self.net(image, self.is_training)

        if self.is_training:
            with tf.name_scope('Optimiser'):
                self.optimiser = tf.train.AdamOptimizer(
                    learning_rate=self.action_param.lr)
            loss_func = LossFunction(
                n_class=self.segmentation_param.num_classes,
                loss_type=self.action_param.loss_type)
            data_loss = loss_func(
                prediction=net_out,
                ground_truth=data_dict.get('label', None),
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
            grads = self.optimiser.compute_gradients(loss)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
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

    def interpret_output(self, batch_output):
        if not self.is_training:
            return self.output_decoder.decode_batch(
                batch_output['window'], batch_output['location'])
        else:
            return True
