# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.channel_sparse_convolution import ChannelSparseConvolutionalLayer
from niftynet.layer.bn import BNLayer

from niftynet.network.base_net import BaseNet
from niftynet.utilities.util_common import look_up_operations
from niftynet.io.misc_io import image3_axial


class DenseVNet(BaseNet):
    """
    implementation of Dense-V-Net:
      Gibson et al., "Automatic multi-organ segmentation
      on abdominal CT with dense V-networks"
    """

    def __init__(self,
                 num_classes,
                 hyperparameters={},
                 architecture_parameters={},
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='DenseVNet'):

        super(DenseVNet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.hyperparameters = \
            {'prior_size': 12,
             'p_channels_selected': .5,
             'n_dense_channels': (4, 8, 16),
             'n_seg_channels': (12, 24, 24),
             'n_input_channels': (24, 24, 24),
             'dilation_rates': ([1] * 5, [1] * 10, [1] * 10),
             # Note dilation rates are not yet supported
             'final_kernel': 3
             }
        self.hyperparameters.update(hyperparameters)
        if any([d != 1 for ds in self.hyperparameters['dilation_rates']
                for d in ds]):
            raise NotImplementedError(
                'Dilated convolutions are not yet implemented')
        self.architecture_parameters = \
            {'use_bdo': False,
             'use_prior': False,
             'use_dense_connections': True,
             'use_coords': False,
             }
        self.architecture_parameters.update(architecture_parameters)
        if self.architecture_parameters['use_dense_connections'] is False:
            raise NotImplementedError(
                'Non-dense connections are not yet implemented')
        if self.architecture_parameters['use_coords'] is True:
            raise NotImplementedError(
                'Image coordinate augmentation is not yet implemented')

    def layer_op(self, input_tensor, is_training, layer_id=-1):
        hp = self.hyperparameters
        channel_dim = len(input_tensor.get_shape()) - 1
        input_size = input_tensor.get_shape().as_list()
        spatial_rank = len(input_size) - 2

        modulo = 2 ** (len(hp['dilation_rates']))
        assert layer_util.check_spatial_dims(input_tensor,
                                             lambda x: x % modulo == 0)

        downsample_channels = list(hp['n_input_channels'][1:]) + [None]
        v_params = zip(hp['n_dense_channels'],
                       hp['n_seg_channels'],
                       downsample_channels,
                       hp['dilation_rates'],
                       range(len(downsample_channels)))

        downsampled_img = BNLayer()(tf.nn.avg_pool3d(input_tensor,
                                                     [1] + [3] * spatial_rank + [1],
                                                     [1] + [2] * spatial_rank + [1],
                                                     'SAME'), is_training=is_training)
        all_segmentation_features = [downsampled_img]
        output_shape = downsampled_img.get_shape().as_list()[1:-1]
        initial_features = ConvolutionalLayer(
            hp['n_input_channels'][0],
            kernel_size=5, stride=2)(input_tensor, is_training=is_training)

        down = tf.concat([downsampled_img, initial_features], channel_dim)
        for dense_ch, seg_ch, down_ch, dil_rate, idx in v_params:
            sd = DenseFeatureStackBlockWithSkipAndDownsample(
                dense_ch,
                3,
                dil_rate,
                seg_ch,
                down_ch,
                self.architecture_parameters['use_bdo'],
                acti_func='relu')
            skip, down = sd(down,
                            is_training=is_training,
                            keep_prob=hp['p_channels_selected'])
            all_segmentation_features.append(image_resize(skip, output_shape))
        segmentation = ConvolutionalLayer(
            self.num_classes,
            kernel_size=hp['final_kernel'],
            with_bn=False,
            with_bias=True)(tf.concat(all_segmentation_features, channel_dim),
                            is_training=is_training)
        if self.architecture_parameters['use_prior']:
            segmentation = segmentation + \
                           SpatialPriorBlock([12] * spatial_rank, output_shape)
        segmentation = image_resize(segmentation, input_size[1:-1])
        image3_axial('seg', tf.nn.softmax(segmentation)[:, :, :, :, 1:] * 255.,
                     3, [tf.GraphKeys.SUMMARIES])
        image3_axial('img',
                     tf.minimum(255., tf.maximum(0.,
                                                 (tf.to_float(downsampled_img) / 2. + 1.) * 127.)),
                     3, [tf.GraphKeys.SUMMARIES])
        return segmentation


def image_resize(image, output_size):
    input_size = image.get_shape().as_list()
    spatial_rank = len(input_size) - 2
    if all([o == i for o, i in zip(output_size, input_size[1:-1])]):
        return image
    if spatial_rank == 2:
        return tf.image.resize_images(image, output_size)
    first_reshape = tf.reshape(image, input_size[0:3] + [-1])
    first_resize = tf.image.resize_images(first_reshape, output_size[0:2])
    second_shape = input_size[:1] + [output_size[0] * output_size[1], input_size[3], -1]
    second_reshape = tf.reshape(first_resize, second_shape)
    second_resize = tf.image.resize_images(second_reshape,
                                           [second_shape[1], output_size[2]])
    final_shape = input_size[:1] + output_size + input_size[-1:]
    return tf.reshape(second_resize, final_shape)


class SpatialPriorBlock(TrainableLayer):
    def __init__(self,
                 prior_shape,
                 output_shape,
                 name='spatial_prior_block'):
        super(SpatialPriorBlock, self).__init__(name=name)
        self.prior_shape = prior_shape
        self.output_shape = output_shape

    def layer_op(self):
        # The internal representation is probabilities so
        # that resampling makes sense
        prior = tf.get_variable('prior',
                                shape=self.prior_shape,
                                initializer=tf.constant_initializer(1))
        return tf.log(image_resize(prior, self.output_shape))


class DenseFeatureStackBlock(TrainableLayer):
    def __init__(self,
                 n_dense_channels,
                 kernel_size,
                 dilation_rates,
                 use_bdo,
                 name='dense_feature_stack_block',
                 **kwargs):
        super(DenseFeatureStackBlock, self).__init__(name=name)
        self.n_dense_channels = n_dense_channels
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.use_bdo = use_bdo
        self.kwargs = kwargs

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):
        channel_dim = len(input_tensor.get_shape()) - 1
        stack = [input_tensor]
        input_mask = tf.ones([input_tensor.get_shape().as_list()[-1]]) > 0
        for idx, d in enumerate(self.dilation_rates):
            if idx == len(self.dilation_rates) - 1:
                keep_prob = None  # no dropout on last layer of the stack
            if self.use_bdo:
                conv = ChannelSparseConvolutionalLayer(
                    self.n_dense_channels,
                    kernel_size=self.kernel_size,
                    **self.kwargs)
                conv, new_input_mask = conv(tf.concat(stack, channel_dim),
                                            input_mask=input_mask,
                                            is_training=is_training,
                                            keep_prob=keep_prob)
                input_mask = tf.concat([input_mask, new_input_mask], 0)
            else:
                conv = ConvolutionalLayer(self.n_dense_channels,
                                          kernel_size=self.kernel_size,
                                          **self.kwargs)
                conv = conv(tf.concat(stack, channel_dim),
                            is_training=is_training,
                            keep_prob=keep_prob)
            stack.append(conv)
        return stack


class DenseFeatureStackBlockWithSkipAndDownsample(TrainableLayer):
    def __init__(self,
                 n_dense_channels,
                 kernel_size,
                 dilation_rates,
                 n_seg_channels,
                 n_downsample_channels,
                 use_bdo,
                 name='dense_feature_stack_block',
                 **kwargs):
        super(DenseFeatureStackBlockWithSkipAndDownsample,
              self).__init__(name=name)
        self.n_dense_channels = n_dense_channels
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.n_seg_channels = n_seg_channels
        self.n_downsample_channels = n_downsample_channels
        self.use_bdo = use_bdo
        self.kwargs = kwargs

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):
        stack = DenseFeatureStackBlock(
            self.n_dense_channels,
            self.kernel_size,
            self.dilation_rates,
            self.use_bdo,
            **self.kwargs)(input_tensor,
                           is_training=is_training,
                           keep_prob=keep_prob)
        all_features = tf.concat(stack, len(input_tensor.get_shape()) - 1)
        seg = ConvolutionalLayer(
            self.n_seg_channels,
            kernel_size=self.kernel_size,
            **self.kwargs)(all_features,
                           is_training=is_training,
                           keep_prob=keep_prob)
        if self.n_downsample_channels is None:
            down = None
        else:
            down = ConvolutionalLayer(
                self.n_downsample_channels,
                kernel_size=self.kernel_size,
                stride=2,
                **self.kwargs)(all_features,
                               is_training=is_training,
                               keep_prob=keep_prob)
        return seg, down
