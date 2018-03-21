# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from collections import namedtuple
import abc

import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.channel_sparse_convolution \
    import ChannelSparseConvolutionalLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.spatial_transformer import ResamplerLayer
from niftynet.layer.grid_warper import AffineGridWarperLayer

from niftynet.network.base_net import BaseNet
from niftynet.io.misc_io import image3_axial


# Create a structure with all the fields of a DenseVNet network
DenseVNetDesc = namedtuple(
    'DenseVNetParts',
    ['initial_bn', 'initial_conv', 'dense_vblocks', 'seg_layer']
)


class DenseVNet(BaseNet):
    """
    implementation of Dense-V-Net:
       Gibson et al.
       Automatic multi-organ segmentation on abdominal CT with dense V-networks

    ### Diagram

    DFS = Dense Feature Stack Block

    - Initial image is first downsampled to a given size.
    - Each DFS+SD outputs a skip link + a downsampled output.
    - All outputs are upscaled to the initial downsampled size.
    - If initial prior is given add it to the output prediction.

    Input
      |
      --[ DFS ]-----------------------[ Conv ]------------[ Conv ]------[+]-->
           |                                       |  |              |
           -----[ DFS ]---------------[ Conv ]------  |              |
                   |                                  |              |
                   -----[ DFS ]-------[ Conv ]---------              |
                                                          [ Prior ]---

    The layer DenseFeatureStackBlockWithSkipAndDownsample layer implements
    [DFS + Conv + Downsampling] in a single module, and outputs 2 elements:
        - Skip layer:          [ DFS + Conv]
        - Downsampled output:  [ DFS + Down]

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

    def create_network(self):
        hyper = self.hyperparameters

        # Initial Convolution
        net_initial_conv = ConvolutionalLayer(
            hyper['n_input_channels'][0],
            kernel_size=5, stride=2
        )

        # Dense Block Params
        downsample_channels = list(hyper['n_input_channels'][1:]) + [None]
        num_blocks = len(hyper["n_dense_channels"])
        use_bdo = self.architecture_parameters['use_bdo']

        # Create DenseBlocks
        net_dense_vblocks = []

        for idx in range(num_blocks):
            dense_ch = hyper["n_dense_channels"][idx]  # Num dense channels
            seg_ch = hyper["n_seg_channels"][idx]      # Num segmentation ch
            down_ch = downsample_channels[idx]      # Num of downsampling ch
            dil_rate = hyper["dilation_rates"][idx]    # Dilation rate

            # Dense feature block
            dblock = DenseFeatureStackBlockWithSkipAndDownsample(
                dense_ch, 3, dil_rate, seg_ch, down_ch, use_bdo,
                acti_func='relu'
            )

            net_dense_vblocks.append(dblock)

        # Segmentation
        net_seg_layer = ConvolutionalLayer(
            self.num_classes, kernel_size=hyper['final_kernel'],
            with_bn=False, with_bias=True
        )

        return DenseVNetDesc(initial_bn=BNLayer(),
                             initial_conv=net_initial_conv,
                             dense_vblocks=net_dense_vblocks,
                             seg_layer=net_seg_layer)

    def downsample_input(self, input_tensor, n_spatial_dims):
        # Initial downsampling params
        d_size1 = (1,) + (3,) * n_spatial_dims + (1,)
        d_size2 = (1,) + (2,) * n_spatial_dims + (1,)

        # Downsample input
        if n_spatial_dims == 2:
            return tf.nn.avg_pool(input_tensor, d_size1, d_size2, 'SAME')
        elif n_spatial_dims == 3:
            return tf.nn.avg_pool3d(input_tensor, d_size1, d_size2, 'SAME')
        else:
            raise NotImplementedError(
                'Downsampling only supports 2D and 3D images')

    def layer_op(self, input_tensor, is_training, layer_id=-1):
        hyper = self.hyperparameters

        # Initialize DenseVNet network layers
        net = self.create_network()

        #
        # Parameter handling
        #

        # Shape and dimension variable shortcuts
        channel_dim = len(input_tensor.shape) - 1
        input_size = input_tensor.shape.as_list()
        spatial_size = input_size[1:-1]
        n_spatial_dims = input_tensor.shape.ndims - 2

        # Quick access to hyperparams
        pkeep = hyper['p_channels_selected']

        # Validate input dimension with dilation rates
        modulo = 2 ** (len(hyper['dilation_rates']))
        assert layer_util.check_spatial_dims(input_tensor,
                                             lambda x: x % modulo == 0)

        #
        # Augmentation + Downsampling + Initial Layers
        #

        # On the fly data augmentation
        if is_training and hyper['augmentation_scale'] > 0:
            if n_spatial_dims == 2:
                augmentation_class = Affine2DAugmentationLayer
            elif n_spatial_dims == 3:
                augmentation_class = Affine3DAugmentationLayer
            else:
                raise NotImplementedError(
                    'Affine augmentation only supports 2D and 3D images')

            augment_layer = augmentation_class(hyper['augmentation_scale'],
                                               'LINEAR', 'ZERO')
            input_tensor = augment_layer(input_tensor)

        # Variable storing all intermediate results -- VLinks
        all_segmentation_features = []

        # Downsample input to the network
        down_tensor = self.downsample_input(input_tensor, n_spatial_dims)
        downsampled_img = net.initial_bn(down_tensor, is_training=is_training)

        # Add initial downsampled image VLink
        all_segmentation_features.append(downsampled_img)

        # All results should match the downsampled input's shape
        output_shape = downsampled_img.shape.as_list()[1:-1]

        init_features = net.initial_conv(input_tensor, is_training=is_training)

        #
        # Dense VNet Main Block
        #

        # `down` will handle the input of each Dense VNet block
        # Initialize it by stacking downsampled image and initial conv features
        down = tf.concat([downsampled_img, init_features], channel_dim)

        # Process Dense VNet Blocks
        for dblock in net.dense_vblocks:
            # Get skip layer and activation output
            skip, down = dblock(down, is_training=is_training, keep_prob=pkeep)

            # Resize skip layer to original shape and add VLink
            skip = image_resize(skip, output_shape)
            all_segmentation_features.append(skip)

        # Concatenate all intermediate skip layers
        inter_results = tf.concat(all_segmentation_features, channel_dim)

        # Initial segmentation output
        seg_output = net.seg_layer(inter_results, is_training=is_training)

        #
        # Dense VNet End - Now postprocess outputs
        #

        # Refine segmentation with prior if any
        if self.architecture_parameters['use_prior']:
            xyz_prior = SpatialPriorBlock([12] * n_spatial_dims, output_shape)
            seg_output += xyz_prior

        # Invert augmentation if any
        if is_training and hyper['augmentation_scale'] > 0:
            inverse_aug = augment_layer.inverse()
            seg_output = inverse_aug(seg_output)

        # Resize output to original size
        seg_output = image_resize(seg_output, spatial_size)

        # Segmentation results
        seg_argmax = tf.to_float(tf.expand_dims(tf.argmax(seg_output, -1), -1))
        seg_summary = seg_argmax * (255. / self.num_classes - 1)

        # Image Summary
        norm_axes = list(range(1, n_spatial_dims+1))
        mean, var = tf.nn.moments(input_tensor, axes=norm_axes, keep_dims=True)
        timg = tf.to_float(input_tensor - mean) / (tf.sqrt(var) * 2.)
        timg = (timg + 1.) * 127.
        single_channel = tf.reduce_mean(timg, axis=-1, keep_dims=True)
        img_summary = tf.minimum(255., tf.maximum(0., single_channel))
        if n_spatial_dims == 2:
            tf.summary.image(
                tf.get_default_graph().unique_name('imgseg'),
                tf.concat([img_summary, seg_summary], 1),
                5, [tf.GraphKeys.SUMMARIES])
        elif n_spatial_dims == 3:
            # Show summaries
            image3_axial(
                tf.get_default_graph().unique_name('imgseg'),
                tf.concat([img_summary, seg_summary], 1),
                5, [tf.GraphKeys.SUMMARIES])
        else:
            raise NotImplementedError(
                'Image Summary only supports 2D and 3D images')

        return seg_output


def image_resize(image, output_size):
    input_size = image.shape.as_list()
    spatial_rank = len(input_size) - 2
    if all([o == i for o, i in zip(output_size, input_size[1:-1])]):
        return image
    if spatial_rank == 2:
        return tf.image.resize_images(image, output_size)
    first_reshape = tf.reshape(image, input_size[0:3] + [-1])
    first_resize = tf.image.resize_images(first_reshape, output_size[0:2])
    second_shape = input_size[:1] + [output_size[0] * output_size[1],
                                     input_size[3], -1]
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


DenseFSBlockDesc = namedtuple('DenseFSDesc', ['conv_layers'])


class DenseFeatureStackBlock(TrainableLayer):
    """
    Dense Feature Stack Block

    - Stack is initialized with the input from above layers.
    - Iteratively the output of convolution layers is added to the stack.
    - Each sequential convolution is performed over all the previous stacked
      channels.

    Diagram example:

        stack = [Input]
        stack = [stack, conv(stack)]
        stack = [stack, conv(stack)]
        stack = [stack, conv(stack)]
        ...
        Output = [stack, conv(stack)]

    """

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

    def create_block(self):
        net_conv_layers = []

        for _ in self.dilation_rates:
            if self.use_bdo:
                conv = ChannelSparseConvolutionalLayer(
                    self.n_dense_channels, kernel_size=self.kernel_size,
                    **self.kwargs
                )
            else:
                conv = ConvolutionalLayer(self.n_dense_channels,
                                          kernel_size=self.kernel_size,
                                          **self.kwargs)
            net_conv_layers.append(conv)

        return DenseFSBlockDesc(conv_layers=net_conv_layers)

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):
        # Initialize FeatureStackBlocks
        block = self.create_block()

        stack = [input_tensor]
        channel_dim = len(input_tensor.shape) - 1
        input_mask = tf.ones([input_tensor.shape.as_list()[-1]]) > 0

        # Stack all convolution outputs
        for idx, conv in enumerate(block.conv_layers):
            if idx == len(self.dilation_rates) - 1:
                keep_prob = None  # no dropout on last layer of the stack

            if self.use_bdo:
                conv, new_input_mask = conv(tf.concat(stack, channel_dim),
                                            input_mask=input_mask,
                                            is_training=is_training,
                                            keep_prob=keep_prob)
                input_mask = tf.concat([input_mask, new_input_mask], 0)
            else:
                conv = conv(tf.concat(stack, channel_dim),
                            is_training=is_training,
                            keep_prob=keep_prob)

            stack.append(conv)

        return stack


DenseSDBlockDesc = namedtuple('DenseSDBlock', ['dense_fstack', 'conv', 'down'])


class DenseFeatureStackBlockWithSkipAndDownsample(TrainableLayer):
    """
    Dense Feature Stack with Skip Layer and Downsampling

    - Downsampling is done through strided convolution.

    ---[ DenseFeatureStackBlock ]----------[ Conv ]------- Skip layer
                                      |
                                      -------------------- Downsampled Output

    See DenseFeatureStackBlock for more info.
    """

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

    def create_block(self):
        net_dense_fstack = DenseFeatureStackBlock(
            self.n_dense_channels, self.kernel_size, self.dilation_rates,
            self.use_bdo, **self.kwargs
        )

        net_conv = ConvolutionalLayer(
            self.n_seg_channels, kernel_size=self.kernel_size, **self.kwargs
        )

        net_down = None
        if self.n_downsample_channels is not None:
            net_down = ConvolutionalLayer(self.n_downsample_channels,
                                          kernel_size=self.kernel_size,
                                          stride=2, **self.kwargs)

        return DenseSDBlockDesc(dense_fstack=net_dense_fstack,
                                conv=net_conv, down=net_down)

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):
        # Current block model
        block = self.create_block()

        # Dense Feature Stack
        stack = block.dense_fstack(input_tensor, is_training=is_training,
                                   keep_prob=keep_prob)

        all_features = tf.concat(stack, len(input_tensor.shape) - 1)

        # Output Convolution
        seg = block.conv(all_features, is_training=is_training,
                         keep_prob=keep_prob)

        # Downsample if needed
        down = None
        if block.down is not None:
            down = block.down(all_features, is_training=is_training,
                              keep_prob=keep_prob)
        return seg, down


class AffineAugmentationLayer(TrainableLayer):
    """ This layer applies a small random (per-iteration) affine
    transformation to an image. The distribution of transformations
    generally results in scaling the image up, with minimal sampling
    outside the original image."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, scale, interpolation,
                 boundary, transform_func=None,
                 name='AffineAugmentation'):
        """"
        scale denotes how extreme the perturbation is, with 1. meaning
            no perturbation and 0.5 giving larger perturbations.
        interpolation denotes the image value interpolation used by
            the resampling
        boundary denotes the boundary handling used by the resampling
        transform_func should be a function returning a relative
        transformation (mapping <-1..1,-1..1,-1..1> to <-1..1,-1..1,-1..1>
        or <-1..1,-1..1> to <-1..1,-1..1>)"""
        super(AffineAugmentationLayer, self).__init__(name=name)
        self.scale = scale
        if transform_func is None:
            self.transform_func = self.random_transform
        else:
            self.transform_func = transform_func
        self._transform = None
        self.interpolation = interpolation
        self.boundary = boundary

    def random_transform(self, batch_size):
        if self._transform is None:
            corners_ = self.get_corners()

            _batch_ones = tf.ones([batch_size, len(corners_[0]), 1])

            corners = tf.tile(corners_, [batch_size, 1, 1])
            random_size = [batch_size, len(corners_[0]), len(corners_[0][0])]
            random_scale = tf.random_uniform(random_size, 0, self.scale)
            corners2 = corners * (1 - random_scale)
            corners_homog = tf.concat([corners, _batch_ones], 2)
            corners2_homog = tf.concat([corners2, _batch_ones], 2)

            _transform = tf.matrix_solve_ls(corners_homog, corners2_homog)
            self._transform = tf.transpose(_transform, [0, 2, 1])

        return self._transform

    def inverse_transform(self, batch_size):
        return tf.matrix_inverse(self.transform_func(batch_size))

    def layer_op(self, input_tensor):
        size = input_tensor.shape.as_list()
        grid_warper = AffineGridWarperLayer(size[1:-1],
                                            size[1:-1])

        resampler = ResamplerLayer(interpolation=self.interpolation,
                                   boundary=self.boundary)

        relative_transform = self.transform_func(size[0])
        to_relative = tf.tile(self.get_tfm_to_relative(size), [size[0], 1, 1])

        from_relative = tf.matrix_inverse(to_relative)
        voxel_transform = tf.matmul(from_relative,
                                    tf.matmul(relative_transform, to_relative))
        dims = self.spatial_dims
        warp_parameters = tf.reshape(voxel_transform[:, 0:dims, 0:dims + 1],
                                     [size[0], dims * (dims + 1)])
        grid = grid_warper(warp_parameters)
        return resampler(input_tensor, grid)

    def inverse(self, interpolation=None, boundary=None):
        if interpolation is None:
            interpolation = self.interpolation
        if boundary is None:
            boundary = self.boundary

        return self.__class__(self.scale,
                              interpolation,
                              boundary,
                              self.inverse_transform)

    @abc.abstractproperty
    def spatial_dims(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_corners(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_tfm_to_relative(self):
        raise NotImplementedError

class Affine2DAugmentationLayer(AffineAugmentationLayer):
    """ Specialization of AffineAugmentationLayer for 2D coordinates """
    spatial_dims = 2
    def get_corners(self):
        return [[[-1., -1.],
                 [-1., 1.],
                 [1., -1.],
                 [1., 1.]]]

    def get_tfm_to_relative(self, size):
        return [[[2./(size[1]-1), 0., -1.],
                 [0., 2. / (size[2] - 1), -1.],
                 [0., 0., 1.]]]

class Affine3DAugmentationLayer(AffineAugmentationLayer):
    """ Specialization of AffineAugmentationLayer for 3D coordinates """
    spatial_dims = 3
    def get_corners(self):
        return [[[-1., -1., -1.],
                 [-1., -1., 1.],
                 [-1., 1., -1.],
                 [-1., 1., 1.],
                 [1., -1., -1.],
                 [1., -1., 1.],
                 [1., 1., -1.],
                 [1., 1., 1.]]]
    def get_tfm_to_relative(self, size):
        return [[[2./(size[1]-1), 0., 0., -1.],
                 [0., 2. / (size[2] - 1), 0., -1.],
                 [0., 0., 2. / (size[3] - 1), -1.],
                 [0., 0., 0., 1.]]]
