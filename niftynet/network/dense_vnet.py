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

    __hyper_params__ = dict(
        prior_size=12,
        p_channels_selected=0.5,
        n_dense_channels=(4, 8, 16),
        n_seg_channels=(12, 24, 24),
        n_input_channels=(24, 24, 24),
        dilation_rates=([1] * 5, [1] * 10, [1] * 10),
        final_kernel=3,
        augmentation_scale=0.1
    )

    __net_params__ = dict(
        use_bdo=False,
        use_prior=False,
        use_dense_connections=True,
        use_coords=False
    )

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

        # Override default Hyperparameters
        self.hyperparameters = dict(self.__hyper_params__)
        self.hyperparameters.update(hyperparameters)

        # Check for dilation rates
        if any([d != 1 for ds in self.hyperparameters['dilation_rates']
                for d in ds]):
            raise NotImplementedError(
                'Dilated convolutions are not yet implemented')

        # Override default architectural parameters
        self.architecture_parameters = dict(self.__net_params__)
        self.architecture_parameters.update(architecture_parameters)

        # Check available modes
        if self.architecture_parameters['use_dense_connections'] is False:
            raise NotImplementedError(
                'Non-dense connections are not yet implemented')
        if self.architecture_parameters['use_coords'] is True:
            raise NotImplementedError(
                'Image coordinate augmentation is not yet implemented')

    def layer_op(self, input_tensor, is_training, layer_id=-1):
        hp = self.hyperparameters

        #
        # Parameter handling
        #

        # Shape and dimension variable shortcuts
        channel_dim = len(input_tensor.get_shape()) - 1
        input_size = input_tensor.get_shape().as_list()
        spatial_size = input_size[1:-1]
        n_spatial_dims = len(input_size) - 2

        # Quick access to hyperparams
        num_blocks = len(hp["n_dense_channels"])
        use_bdo = self.architecture_parameters['use_bdo']
        pkeep = hp['p_channels_selected']
        final_kernel_size = hp['final_kernel']

        # Validate input dimension with dilation rates
        modulo = 2 ** (len(hp['dilation_rates']))
        assert layer_util.check_spatial_dims(input_tensor,
                                             lambda x: x % modulo == 0)

        #
        # Preprocessing + Initial Layers
        #

        # On the fly data augmentation
        if is_training and hp['augmentation_scale'] > 0:
            aug_scale = hp['augmentation_scale']
            augment_layer = Affine3DAugmentationLayer(aug_scale, 'LINEAR', 'ZERO')
            input_tensor = augment_layer(input_tensor)

        # Variable storing all intermediate results
        all_segmentation_features = []

        # Initial downsampling params
        downsample_channels = list(hp['n_input_channels'][1:]) + [None]
        d_size1 = (1,) + (3,) * n_spatial_dims + (1,)
        d_size2 = (1,) + (2,) * n_spatial_dims + (1,)

        # Downsample input
        init_bnlayer = BNLayer()
        down_tensor = tf.nn.avg_pool3d(input_tensor, d_size1, d_size2, 'SAME')
        downsampled_img = init_bnlayer(down_tensor, is_training=is_training)

        # Add initial downsampled image as intermediate result
        all_segmentation_features.append(downsampled_img)

        # All results should match the downsampled input's shape
        output_shape = downsampled_img.get_shape().as_list()[1:-1]

        # Initial Convolution
        initial_conv = ConvolutionalLayer(
            hp['n_input_channels'][0],
            kernel_size=5, stride=2
        )

        initial_features = initial_conv(input_tensor, is_training=is_training)

        #
        # Dense VNet Main Block
        #

        # `down` will handle the input of each Dense VNet block
        # Initialize it by stacking downsampled image and initial conv features
        down = tf.concat([downsampled_img, initial_features], channel_dim)

        # Process Dense VNet Blocks
        for idx in range(num_blocks):
            dense_ch = hp["n_dense_channels"][idx]  # Number or dense channels
            seg_ch = hp["n_seg_channels"][idx]      # Number of segmentation ch
            down_ch = downsample_channels[idx]      # Number of downsampling ch
            dil_rate = hp["dilation_rates"][idx]    # Dilation rate

            # Dense feature block
            dblock = DenseFeatureStackBlockWithSkipAndDownsample(
                dense_ch, 3, dil_rate, seg_ch, down_ch, use_bdo,
                acti_func='relu'
            )

            # Get skip layer and activation output
            skip, down = dblock(down, is_training=is_training, keep_prob=pkeep)

            # Resize skip layer to original shape and store it
            skip = image_resize(skip, output_shape)
            all_segmentation_features.append(skip)

        # Concatenate all intermediate skip layers
        inter_resutls = tf.concat(all_segmentation_features, channel_dim)

        # Get segmentation layer
        segmentation_layer = ConvolutionalLayer(
            self.num_classes, kernel_size=final_kernel_size,
            with_bn=False, with_bias=True
        )

        # Initial segmentation output
        seg_output = segmentation_layer(inter_resutls, is_training=is_training)

        #
        # Dense VNet End - Now postprocess outputs
        #

        # Refine segmentation with prior if any
        if self.architecture_parameters['use_prior']:
            xyz_prior = SpatialPriorBlock([12] * n_spatial_dims, output_shape)
            seg_output += xyz_prior

        # Invert augmentation if any
        if is_training and hp['augmentation_scale'] > 0:
            inverse_aug = aug.inverse()
            seg_output = inverse_aug(seg_output)

        # Resize output to original size
        seg_output = image_resize(seg_output, spatial_size)

        # Segmentation results
        seg_argmax = tf.to_float(tf.expand_dims(tf.argmax(seg_output, -1), -1))
        seg_summary = seg_argmax * (255. / self.num_classes - 1)

        # Image Summary
        m, v = tf.nn.moments(input_tensor, axes=[1, 2, 3], keep_dims=True)
        timg = (tf.to_float(input_tensor - m) / (tf.sqrt(v) * 2.) + 1.) * 127.
        img_summary = tf.minimum(255., tf.maximum(0., timg))

        # Show summaries
        image3_axial('imgseg', tf.concat([img_summary, seg_summary], 1),
                     5, [tf.GraphKeys.SUMMARIES])

        return seg_output


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
        sz = input_tensor.get_shape().as_list()
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
