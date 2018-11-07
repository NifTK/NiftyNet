# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from collections import namedtuple

import tensorflow as tf

from niftynet.io.misc_io import image3_axial
from niftynet.layer import layer_util
from niftynet.layer.affine_augmentation import AffineAugmentationLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.channel_sparse_convolution \
    import ChannelSparseConvolutionalLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.linear_resize import LinearResizeLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.network.base_net import BaseNet

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
            seg_ch = hyper["n_seg_channels"][idx]  # Num segmentation ch
            down_ch = downsample_channels[idx]  # Num of downsampling ch
            dil_rate = hyper["dilation_rates"][idx]  # Dilation rate

            # Dense feature block
            dblock = DenseFeatureStackBlockWithSkipAndDownsample(
                dense_ch, 3, dil_rate, seg_ch, down_ch, use_bdo,
                acti_func='relu'
            )

            net_dense_vblocks.append(dblock)

        # Segmentation
        net_seg_layer = ConvolutionalLayer(
            self.num_classes, kernel_size=hyper['final_kernel'],
            bn_type=None, with_bias=True
        )

        return DenseVNetDesc(initial_bn=BNLayer(),
                             initial_conv=net_initial_conv,
                             dense_vblocks=net_dense_vblocks,
                             seg_layer=net_seg_layer)

    def layer_op(self,
                 input_tensor,
                 is_training=True,
                 layer_id=-1,
                 keep_prob=0.5,
                 **unused_kwargs):
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

        # Validate input dimension with dilation rates
        modulo = 2 ** (len(hyper['dilation_rates']))
        assert layer_util.check_spatial_dims(input_tensor,
                                             lambda x: x % modulo == 0)

        #
        # Augmentation + Downsampling + Initial Layers
        #

        # On the fly data augmentation
        augment_layer = None
        if is_training and hyper['augmentation_scale'] > 0:
            augmentation_class = AffineAugmentationLayer
            augment_layer = augmentation_class(
                hyper['augmentation_scale'], 'LINEAR', 'ZERO')
            input_tensor = augment_layer(input_tensor)

        # Variable storing all intermediate results -- VLinks
        all_segmentation_features = []

        # Downsample input to the network
        ave_downsample_layer = DownSampleLayer(
            func='AVG', kernel_size=3, stride=2)
        down_tensor = ave_downsample_layer(input_tensor)
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
            skip, down = dblock(down,
                                is_training=is_training,
                                keep_prob=keep_prob)

            # Resize skip layer to original shape and add VLink
            skip = LinearResizeLayer(output_shape)(skip)
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
        if is_training and hyper['augmentation_scale'] > 0 \
                and augment_layer is not None:
            inverse_aug = augment_layer.inverse()
            seg_output = inverse_aug(seg_output)

        # Resize output to original size
        seg_output = LinearResizeLayer(spatial_size)(seg_output)

        # Segmentation results
        seg_argmax = tf.to_float(tf.expand_dims(tf.argmax(seg_output, -1), -1))
        seg_summary = seg_argmax * (255. / self.num_classes - 1)

        # Image Summary
        norm_axes = list(range(1, n_spatial_dims + 1))
        mean, var = tf.nn.moments(input_tensor, axes=norm_axes, keep_dims=True)
        timg = tf.to_float(input_tensor - mean) / (tf.sqrt(var) * 2.)
        timg = (timg + 1.) * 127.
        single_channel = tf.reduce_mean(timg, -1, True)
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
        return tf.log(LinearResizeLayer(self.output_shape)(prior))


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

    def layer_op(self, input_tensor, is_training=True, keep_prob=None):
        # Initialize FeatureStackBlocks
        block = self.create_block()

        stack = [input_tensor]
        channel_dim = len(input_tensor.shape) - 1
        n_channels = input_tensor.shape.as_list()[-1]
        input_mask = tf.ones([n_channels]) > 0

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

        if self.use_bdo:  # unmask the conv channels
            # modify the returning stack by:
            # 1. Removing the input of the DFS from the stack
            # 2. Unmasking the stack by filling in zeros
            # see: https://github.com/NifTK/NiftyNet/pull/101

            conv_channels = tf.concat(stack[1:], axis=-1)

            # insert a channel with zeros to be placed
            # where channels were not calculated
            zero_channel = tf.zeros(conv_channels.shape[:-1])
            zero_channel = tf.expand_dims(zero_channel, axis=-1)
            conv_channels = tf.concat([zero_channel, conv_channels], axis=-1)

            # indices to keep
            int_mask = tf.cast(input_mask[n_channels:], tf.int32)
            indices = tf.cumsum(int_mask) * int_mask
            # rearrange stack with zeros where channels were not calculated
            conv_channels = tf.gather(conv_channels, indices, axis=-1)
            stack = [conv_channels]

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

    def layer_op(self, input_tensor, is_training=True, keep_prob=None):
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
