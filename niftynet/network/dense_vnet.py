# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

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

from collections import namedtuple


class DenseVNet(BaseNet):
    """
    ### Description
    implementation of Dense-V-Net:
       Gibson et al., "Automatic multi-organ segmentation on abdominal CT with
       dense V-networks" 2018

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

    ### Constraints
    - Input size has to be divisible by 2*dilation_rates

    """

    """ Default network hyperparameters

    Params:
        prior_size (): size of spatial prior
        n_dense_channels (): num dense channels in each block
        n_seg_channels (): num of segmentation channels
        n_initial_conv_channels (): num of channels in inital convolution
        n_down_channels (): num of downsampling channels
        dilation_rate (): dilation rate of each layer in each vblock
        seg_kernel_size (): kernel size of final conv segmentation
        augmentation_scale (): determines extent of the affine perturbation.
            0.0 gives no perturbation and 1.0 gives the largest perturbation
        use_bdo (): use batch-wise dropout
        use_prior (): use spatial prior
        use_dense_connections (): densely connect layers of each vblock
        use_coords (): use image coordinate augmentation
    """
    __hyper_params__ = dict(
        prior_size=24,
        n_dense_channels=[4, 8, 16],
        n_seg_channels=[12, 24, 24],
        n_initial_conv_channels=24,
        n_down_channels=[24, 24, None],
        dilation_rates=[[1] * 5, [1] * 10, [1] * 10],
        seg_kernel_size=3,
        augmentation_scale=0.1,
        use_bdo=False,
        use_prior=False,
        use_dense_connections=True,
        use_coords=False)

    def __init__(self,
                 num_classes,
                 hyperparams={},
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='DenseVNet'):
        """

        :param num_classes: int, number of channels of output
        :param hyperparams: dictionary, network hyperparameters
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param acti_func: activation function to use
        :param name: layer name
        """

        super(DenseVNet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        # Override default Hyperparameters
        self.hyperparams = self.__hyper_params__
        self.hyperparams.update(hyperparams)

        # Check for dilation rates
        if any([d != 1 for ds in self.hyperparams['dilation_rates']
                for d in ds]):
            raise NotImplementedError(
                'Dilated convolutions are not yet implemented')
        # Check available modes
        if self.hyperparams['use_dense_connections'] is False:
            raise NotImplementedError(
                'Non-dense connections are not yet implemented')
        if self.hyperparams['use_coords'] is True:
            raise NotImplementedError(
                'Image coordinate augmentation is not yet implemented')

    def create_network(self):

        hyperparams = self.hyperparams

        # Create initial convolutional layer
        initial_conv = ConvolutionalLayer(
            hyperparams['n_initial_conv_channels'],
            kernel_size=5,
            stride=2)
        # name='initial_conv')

        # Create dense vblocks
        num_blocks = len(hyperparams["n_dense_channels"])  # Num dense blocks
        dense_ch = hyperparams["n_dense_channels"]
        seg_ch = hyperparams["n_seg_channels"]
        down_ch = hyperparams["n_down_channels"]
        dil_rate = hyperparams["dilation_rates"]
        use_bdo = hyperparams['use_bdo']

        dense_vblocks = []
        for i in range(num_blocks):
            vblock = DenseFeatureStackBlockWithSkipAndDownsample(
                n_dense_channels=dense_ch[i],
                kernel_size=3,
                dilation_rates=dil_rate[i],
                n_seg_channels=seg_ch[i],
                n_down_channels=down_ch[i],
                use_bdo=use_bdo,
                acti_func='relu')
            dense_vblocks.append(vblock)

        # Create final convolutional layer
        final_conv = ConvolutionalLayer(
            self.num_classes,
            kernel_size=hyperparams['seg_kernel_size'],
            feature_normalization=None,
            with_bias=True)
        #  name='final_conv')

        # Create a structure with all the fields of a DenseVNet
        dense_vnet = namedtuple('DenseVNet',
                                ['initial_conv', 'dense_vblocks', 'final_conv'])

        return dense_vnet(initial_conv=initial_conv,
                          dense_vblocks=dense_vblocks,
                          final_conv=final_conv)

    def layer_op(self,
                 input_tensor,
                 is_training=True,
                 layer_id=-1,
                 keep_prob=0.5,
                 **unused_kwargs):
        """

        :param input_tensor: tensor to input to the network, size has to be divisible by 2*dilation_rates
        :param is_training: boolean, True if network is in training mode
        :param layer_id: not in use
        :param keep_prob: double, percentage of nodes to keep for drop-out
        :param unused_kwargs:
        :return: network prediction
        """
        hyperparams = self.hyperparams

        # Validate that dilation rates are compatible with input dimensions
        modulo = 2 ** (len(hyperparams['dilation_rates']))
        assert layer_util.check_spatial_dims(input_tensor,
                                             lambda x: x % modulo == 0)

        # Perform on the fly data augmentation
        if is_training and hyperparams['augmentation_scale'] > 0:
            augment_layer = AffineAugmentationLayer(
                hyperparams['augmentation_scale'], 'LINEAR', 'ZERO')
            input_tensor = augment_layer(input_tensor)

        ###################
        ### Feedforward ###
        ###################

        # Initialize network components
        dense_vnet = self.create_network()

        # Store output feature maps from each component
        feature_maps = []

        # Downsample input to the network
        downsample_layer = DownSampleLayer(func='AVG', kernel_size=3, stride=2)
        downsampled_tensor = downsample_layer(input_tensor)
        bn_layer = BNLayer()
        downsampled_tensor = bn_layer(
            downsampled_tensor, is_training=is_training)
        feature_maps.append(downsampled_tensor)

        # All feature maps should match the downsampled tensor's shape
        feature_map_shape = downsampled_tensor.shape.as_list()[1:-1]

        # Prepare initial input to dense_vblocks
        initial_features = dense_vnet.initial_conv(
            input_tensor, is_training=is_training)
        channel_dim = len(input_tensor.shape) - 1
        down = tf.concat([downsampled_tensor, initial_features], channel_dim)

        # Feed downsampled input through dense_vblocks
        for dblock in dense_vnet.dense_vblocks:
            # Get skip layer and activation output
            skip, down = dblock(down,
                                is_training=is_training,
                                keep_prob=keep_prob)
            # Resize skip layer to original shape and add to feature maps
            skip = LinearResizeLayer(feature_map_shape)(skip)
            feature_maps.append(skip)

        # Merge feature maps
        all_features = tf.concat(feature_maps, channel_dim)

        # Perform final convolution to segment structures
        output = dense_vnet.final_conv(all_features, is_training=is_training)

        ######################
        ### Postprocessing ###
        ######################

        # Get the number of spatial dimensions of input tensor
        n_spatial_dims = input_tensor.shape.ndims - 2

        # Refine segmentation with prior
        if hyperparams['use_prior']:
            spatial_prior_shape = [hyperparams['prior_size']] * n_spatial_dims
            # Prior shape must be 4 or 5 dim to work with linear_resize layer
            # ie to conform to shape=[batch, X, Y, Z, channels]
            prior_shape = [1] + spatial_prior_shape + [1]
            spatial_prior = SpatialPriorBlock(prior_shape, feature_map_shape)
            output += spatial_prior()

        # Invert augmentation
        if is_training and hyperparams['augmentation_scale'] > 0:
            inverse_aug = augment_layer.inverse()
            output = inverse_aug(output)

        # Resize output to original size
        input_tensor_spatial_size = input_tensor.shape.as_list()[1:-1]
        output = LinearResizeLayer(input_tensor_spatial_size)(output)

        # Segmentation summary
        seg_argmax = tf.to_float(tf.expand_dims(tf.argmax(output, -1), -1))
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
            image3_axial(
                tf.get_default_graph().unique_name('imgseg'),
                tf.concat([img_summary, seg_summary], 1),
                5, [tf.GraphKeys.SUMMARIES])
        else:
            raise NotImplementedError(
                'Image Summary only supports 2D and 3D images')

        return output


class SpatialPriorBlock(TrainableLayer):
    def __init__(self,
                 prior_shape,
                 output_shape,
                 name='spatial_prior_block'):

        """

        :param prior_shape: shape of spatial prior
        :param output_shape: target shape for resampling
        :param name: layer name
        """

        super(SpatialPriorBlock, self).__init__(name=name)

        self.prior_shape = prior_shape
        self.output_shape = output_shape

    def layer_op(self):
        """

        :return: spatial prior resampled to the target shape
        """
        # The internal representation is probabilities so
        # that resampling makes sense
        prior = tf.get_variable('prior',
                                shape=self.prior_shape,
                                initializer=tf.constant_initializer(1))
        return tf.log(LinearResizeLayer(self.output_shape)(prior))


class DenseFeatureStackBlock(TrainableLayer):
    """
    Dense Feature Stack Block

    - Stack is initialized with the input from above layers.
    - Iteratively the output of convolution layers is added to the feature stack.
    - Each sequential convolution is performed over all the previous stacked
      channels.

    Diagram example:

        feature_stack = [Input]
        feature_stack = [feature_stack, conv(feature_stack)]
        feature_stack = [feature_stack, conv(feature_stack)]
        feature_stack = [feature_stack, conv(feature_stack)]
        ...
        Output = [feature_stack, conv(feature_stack)]

    """

    def __init__(self,
                 n_dense_channels,
                 kernel_size,
                 dilation_rates,
                 use_bdo,
                 name='dense_feature_stack_block',
                 **kwargs):
        """

        :param n_dense_channels: int, number of dense channels in each block
        :param kernel_size: kernel size for convolutional layers
        :param dilation_rates: dilation rate of each layer in each vblock
        :param use_bdo: boolean, set to True to use batch-wise drop-out
        :param name: tensorflow scope name
        :param kwargs:
        """

        super(DenseFeatureStackBlock, self).__init__(name=name)

        self.n_dense_channels = n_dense_channels
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.use_bdo = use_bdo
        self.kwargs = kwargs

    def create_block(self):
        """

        :return:  dense feature stack block
        """
        dfs_block = []
        for _ in self.dilation_rates:
            if self.use_bdo:
                conv = ChannelSparseConvolutionalLayer(
                    self.n_dense_channels,
                    kernel_size=self.kernel_size,
                    **self.kwargs)
            else:
                conv = ConvolutionalLayer(
                    self.n_dense_channels,
                    kernel_size=self.kernel_size,
                    **self.kwargs)

            dfs_block.append(conv)

        return dfs_block

    def layer_op(self, input_tensor, is_training=True, keep_prob=None):
        """

        :param input_tensor: tf tensor, input to the DenseFeatureStackBlock
        :param is_training: boolean, True if network is in training mode
        :param keep_prob: double, percentage of nodes to keep for drop-out
        :return: feature stack
        """
        # Create dense feature stack block
        dfs_block = self.create_block()
        # Initialize feature stack for block
        feature_stack = [input_tensor]

        # Create initial input mask for batch-wise dropout
        n_channels = input_tensor.shape.as_list()[-1]
        input_mask = tf.ones([n_channels]) > 0

        # Stack convolution outputs
        for i, conv in enumerate(dfs_block):
            # No dropout on last layer of the stack
            if i == len(dfs_block) - 1:
                keep_prob = None

            # Merge feature stack along channel dimension
            channel_dim = len(input_tensor.shape) - 1
            input_features = tf.concat(feature_stack, channel_dim)

            if self.use_bdo:
                output_features, new_input_mask = conv(input_features,
                                                       input_mask=input_mask,
                                                       is_training=is_training,
                                                       keep_prob=keep_prob)
                input_mask = tf.concat([input_mask, new_input_mask], 0)
            else:
                output_features = conv(input_features,
                                       is_training=is_training,
                                       keep_prob=keep_prob)

            feature_stack.append(output_features)

        # Unmask the convolution channels
        if self.use_bdo:
            # Modify the returning feature stack by:
            # 1. Removing the input of the DFS from the feature stack
            # 2. Unmasking the feature stack by filling in zeros
            # see: https://github.com/NifTK/NiftyNet/pull/101

            # Remove input of DFS from the feature stack
            conv_channels = tf.concat(feature_stack[1:], axis=-1)

            # Insert a channel with zeros to be placed
            # where channels were not calculated
            zero_channel = tf.zeros(conv_channels.shape[:-1])
            zero_channel = tf.expand_dims(zero_channel, axis=-1)
            conv_channels = tf.concat([zero_channel, conv_channels], axis=-1)

            # Indices to keep
            int_mask = tf.cast(input_mask[n_channels:], tf.int32)
            indices = tf.cumsum(int_mask) * int_mask

            # Rearrange stack with zeros where channels were not calculated
            conv_channels = tf.gather(conv_channels, indices, axis=-1)
            feature_stack = [conv_channels]

        return feature_stack


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
                 n_down_channels,
                 use_bdo,
                 name='dense_feature_stack_block',
                 **kwargs):
        """

        :param n_dense_channels: int, number of dense channels
        :param kernel_size: kernel size for convolutional layers
        :param dilation_rates: dilation rate of each layer in each vblock
        :param n_seg_channels: int, number of segmentation channels
        :param n_down_channels: int, number of output channels when downsampling
        :param use_bdo: boolean, set to True to use batch-wise drop-out
        :param name: layer name
        :param kwargs:
        """

        super(DenseFeatureStackBlockWithSkipAndDownsample, self).__init__(
            name=name)

        self.n_dense_channels = n_dense_channels
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.n_seg_channels = n_seg_channels
        self.n_down_channels = n_down_channels
        self.use_bdo = use_bdo
        self.kwargs = kwargs

    def create_block(self):
        """

        :return: Dense Feature Stack with Skip Layer and Downsampling block
        """
        dfs_block = DenseFeatureStackBlock(self.n_dense_channels,
                                           self.kernel_size,
                                           self.dilation_rates,
                                           self.use_bdo,
                                           **self.kwargs)

        skip_conv = ConvolutionalLayer(self.n_seg_channels,
                                       kernel_size=self.kernel_size,
                                       # name='skip_conv',
                                       **self.kwargs)

        down_conv = None
        if self.n_down_channels is not None:
            down_conv = ConvolutionalLayer(self.n_down_channels,
                                           kernel_size=self.kernel_size,
                                           stride=2,
                                           #  name='down_conv',
                                           **self.kwargs)

        dfssd_block = namedtuple('DenseSDBlock',
                                 ['dfs_block', 'skip_conv', 'down_conv'])

        return dfssd_block(dfs_block=dfs_block,
                           skip_conv=skip_conv,
                           down_conv=down_conv)

    def layer_op(self, input_tensor, is_training=True, keep_prob=None):
        """

        :param input_tensor: tf tensor, input to the DenseFeatureStackBlock
        :param is_training: boolean, True if network is in training mode
        :param keep_prob: double, percentage of nodes to keep for drop-out
        :return: feature stack after skip convolution, feature stack after downsampling
        """
        # Create dense feature stack block with skip and downsample
        dfssd_block = self.create_block()

        # Feed input through the dense feature stack block
        feature_stack = dfssd_block.dfs_block(input_tensor,
                                              is_training=is_training,
                                              keep_prob=keep_prob)

        # Merge feature stack
        merged_features = tf.concat(feature_stack, len(input_tensor.shape) - 1)

        # Perform skip convolution
        skip_conv = dfssd_block.skip_conv(merged_features,
                                          is_training=is_training,
                                          keep_prob=keep_prob)

        # Downsample if needed
        down_conv = None
        if dfssd_block.down_conv is not None:
            down_conv = dfssd_block.down_conv(merged_features,
                                              is_training=is_training,
                                              keep_prob=keep_prob)

        return skip_conv, down_conv
