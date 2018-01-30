# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.channel_sparse_convolution import ChannelSparseConvolutionalLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.spatial_transformer import ResamplerLayer
from niftynet.layer.grid_warper import AffineGridWarperLayer

from niftynet.network.base_net import BaseNet
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
             'final_kernel': 3,
             'augmentation_scale': .1
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
        if is_training and hp['augmentation_scale']>0:
            aug = Affine3DAugmentationLayer(hp['augmentation_scale'],
                                            'LINEAR','ZERO')
            input_tensor=aug(input_tensor)
        channel_dim = len(input_tensor.get_shape()) - 1
        input_size = input_tensor.shape.as_list()
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
        output_shape = downsampled_img.shape.as_list()[1:-1]
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
        if is_training and hp['augmentation_scale']>0:
            inverse_aug = aug.inverse()
            segmentation = inverse_aug(segmentation)
        segmentation = image_resize(segmentation, input_size[1:-1])
        seg_summary = tf.to_float(tf.expand_dims(tf.argmax(segmentation,-1),-1)) * (255./self.num_classes-1)
        m,v = tf.nn.moments(input_tensor,axes=[1,2,3],keep_dims=True)
        img_summary = tf.minimum(255., tf.maximum(0.,
                         (tf.to_float(input_tensor-m) / (tf.sqrt(v) * 2.) + 1.) * 127.))
        image3_axial('imgseg', tf.concat([img_summary,seg_summary],1) ,
                     5, [tf.GraphKeys.SUMMARIES])
        return segmentation


def image_resize(image, output_size):
    input_size = image.shape.as_list()
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
        input_mask = tf.ones([input_tensor.shape.as_list()[-1]]) > 0
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

class Affine3DAugmentationLayer(TrainableLayer):
    def __init__(self,scale,interpolation,
                 boundary, transform_func=None,
                 name='AffineAugmentation'):
        # transform_func should be a function returning
        # a relative transform (mapping <-1..1,-1..1,-1.1>
        # to <-1..1,-1..1,-1.1>)
        super(Affine3DAugmentationLayer,
              self).__init__(name=name)
        self.scale=scale
        if transform_func is None:
            self.transform_func = self.random_transform
        else:
            self.transform_func = transform_func
        self._transform=None
        self.interpolation = interpolation
        self.boundary = boundary

    def random_transform(self,batch_size):
        if self._transform is None:
            corners = [[[-1.,-1.,-1.],[-1.,-1.,1.],[-1.,1.,-1.],[-1.,1.,1.],[1.,-1.,-1.],[1.,-1.,1.],[1.,1.,-1.],[1.,1.,1.]]]
            corners = tf.tile(corners,[batch_size,1,1])
            corners2 = corners * \
                                   (1-tf.random_uniform([batch_size,8,3],0,self.scale))
            corners_homog = tf.concat([corners,tf.ones([batch_size,8,1])],2)
            corners2_homog = tf.concat([corners2,tf.ones([batch_size,8,1])],2)
            _transform = tf.matrix_solve_ls(corners_homog,corners2_homog)
            self._transform = tf.transpose(_transform,[0,2,1])
        return self._transform

    def inverse_transform(self, batch_size):
        return tf.matrix_inverse(self.transform_func(batch_size))

    def layer_op(self, input_tensor):
        sz = input_tensor.shape.as_list()
        grid_warper = AffineGridWarperLayer(sz[1:-1],
                                            sz[1:-1])

        resampler = ResamplerLayer(interpolation=self.interpolation,
                                   boundary=self.boundary)
        relative_transform = self.transform_func(sz[0])
        to_relative=tf.tile([[[2./(sz[1]-1), 0., 0., -1.],
                              [0., 2. / (sz[2] - 1), 0., -1.],
                              [0., 0., 2. / (sz[3] - 1), -1.],
                              [0., 0., 0., 1.]]],[sz[0],1,1])
        from_relative=tf.matrix_inverse(to_relative)
        voxel_transform = tf.matmul(from_relative,
                                    tf.matmul(relative_transform,to_relative))
        warp_parameters = tf.reshape(voxel_transform[:, 0:3, 0:4],
                                     [sz[0], 12])
        grid = grid_warper(warp_parameters)
        return resampler(input_tensor,grid)

    def inverse(self, interpolation=None, boundary=None):
        if interpolation is None:
            interpolation = self.interpolation
        if boundary is None:
            boundary = self.boundary

        return Affine3DAugmentationLayer(self.scale,
                                       interpolation,
                                       boundary,
                                       self.inverse_transform)
