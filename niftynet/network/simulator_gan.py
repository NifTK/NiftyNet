# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.layer.activation import ActiLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.layer.gan_blocks import BaseDiscriminator
from niftynet.layer.gan_blocks import BaseGenerator
from niftynet.layer.gan_blocks import GANImageBlock


class SimulatorGAN(GANImageBlock):
    """
    ### Description
        implementation of
            Hu et al., "Freehand Ultrasound Image Simulation with Spatially-Conditioned
            Generative Adversarial Networks", MICCAI RAMBO 2017
            https://arxiv.org/abs/1707.05392

    ### Building blocks
    [GENERATOR]         - See ImageGenerator below
    [DISCRIMINATOR]     - See ImageDiscriminator below
    Note: See niftynet.layer.gan_blocks for layer_op

    ### Diagram

    RANDOM NOISE --> [GENERATOR] --> [DISCRIMINATOR] --> fake logits
    TRAINING SET ------------------> [DISCRIMINATOR] --> real logits

    ### Constraints
    """

    def __init__(self, name='simulator_GAN'):
        super(SimulatorGAN, self).__init__(
            generator=ImageGenerator(name='generator'),
            discriminator=ImageDiscriminator(name='discriminator'),
            clip=None,
            name=name)


class ImageGenerator(BaseGenerator):
    """
    ### Description
        implementation of generator from
            Hu et al., "Freehand Ultrasound Image Simulation with Spatially-Conditioned
            Generative Adversarial Networks", MICCAI RAMBO 2017
            https://arxiv.org/abs/1707.05392

    ### Building blocks
    [FC]            - Fully connected layer
    [CONV]          - Convolutional layer, conditioning is concatenated to input before the convolution
                        (if conditioning is used)
                        kernel size = 3, activation = relu
    [UPCONV]        - Upsampling block composed of upsampling deconvolution (stride 2, kernel size = 3,
                        activation = relu) and concatenation of conditioning
    [fCONV]         - Final convolutional layer, with no conditioning. Kernel size = 3, activation = tanh

    ### Diagram

    RANDOM NOISE --> [FC] --> [CONV] --> [UPCONV]x3 --> [fCONV] --> OUTPUT IMAGE

    ### Constraints
    """
    def __init__(self, name):
        """

        :param name: layer name
        """
        super(ImageGenerator, self).__init__(name=name)
        self.initializers = {'w': tf.random_normal_initializer(0, 0.02),
                             'b': tf.constant_initializer(0.001)}
        self.noise_channels_per_layer = 0
        self.with_conditionings = [True, True, True, True, False]

    def layer_op(self, random_source, image_size, conditioning, is_training):
        """

        :param random_source: tensor, random noise to start generation
        :param image_size: output image size
        :param conditioning: tensor, conditioning information (e.g. coordinates in physical space)
        :param is_training: boolean, True if network is in training mode
        :return: tensor, generated image
        """
        keep_prob_ph = 1  # not passed in as a placeholder
        add_noise = self.noise_channels_per_layer
        if conditioning is not None:
            conditioning_channels = conditioning.shape.as_list()[-1]
            conditioning_channels = conditioning_channels + add_noise
        else:
            conditioning_channels = add_noise

        # feature channels design pattern
        ch = [512]
        sz = image_size[:-1]
        for i in range(4):
            # compute output n_feature_channels of i-th layer
            new_ch = ch[-1] + conditioning_channels * self.with_conditionings[i]
            new_ch = round(new_ch / 2)
            ch.append(new_ch)
            # compute output spatial size of i-th layer
            sz = [int(round(spatial_len / 2)) for spatial_len in sz]
        ch.append(1)  # last layer single channel image

        # resizing utilities
        spatial_rank = len(image_size) - 1
        if spatial_rank == 3:
            def resize_func(x, sz):
                sz_x = x.shape.as_list()
                r1 = tf.image.resize_images(
                    tf.reshape(x, sz_x[:3] + [-1]), sz[0:2])
                r2 = tf.image.resize_images(
                    tf.reshape(r1, [sz_x[0], sz[0] * sz[1], sz_x[3], -1]),
                    [sz[0] * sz[1], sz[2]])
                resized_3d = tf.reshape(r2, [sz_x[0]] + sz + [sz_x[-1]])
                return resized_3d
        elif spatial_rank == 2:
            resize_func = tf.image.resize_bilinear

        def concat_cond(x, with_conditioning):
            """

            :param x: tensor, input
            :param with_conditioning: boolean, True if conditioning is to be used
            :return: tensor, concatenation of x and conditioning, with noise addition
            """
            noise = []
            if add_noise:
                feature_shape = x.shape.as_list()[0:-1]
                noise_shape = feature_shape + [add_noise]
                noise = [tf.random_normal(noise_shape, 0.0, 0.1)]

            if with_conditioning and conditioning is not None:
                with tf.name_scope('concat_conditioning'):
                    spatial_shape = x.shape.as_list()[1:-1]
                    resized_cond = resize_func(conditioning, spatial_shape)
                    return tf.concat([x, resized_cond] + noise, axis=-1)
            return x

        def conv(ch, x):
            """
            Generates and applies a convolutional layer with relu as activation, kernel size 3,
            batch norm
            :param ch: int, number of output channels for convolutional layer
            :param x: tensor, input to convolutional layer
            :return: tensor, output of convolutiona layer
            """
            with tf.name_scope('conv'):
                conv_layer = ConvolutionalLayer(
                    n_output_chns=ch,
                    kernel_size=3,
                    feature_normalization='batch',
                    with_bias=False,
                    acti_func='relu',
                    w_initializer=self.initializers['w'])
                return conv_layer(x, is_training=is_training)

        def up(ch, x):
            """
            Performs deconvolution operation with kernel size 3, stride 2, batch norm, and relu
            :param ch: int, number of output channels for deconvolutional layer
            :param x: tensor, input to deconvolutional layer
            :return: tensor, output of deconvolutiona layer
            """
            with tf.name_scope('up'):
                deconv_layer = DeconvolutionalLayer(
                    n_output_chns=ch,
                    kernel_size=3,
                    stride=2,
                    feature_normalization='batch',
                    with_bias=False,
                    acti_func='relu',
                    w_initializer=self.initializers['w'])
                return deconv_layer(x, is_training=is_training)

        def up_block(ch, x, with_conditioning):
            """
            Performs upsampling and concatenation with conditioning
            :param ch: int, number of output channels for deconvolutional layer
            :param x: tensor, input to deconvolutional layer
            :param with_conditioning: boolean, True if conditioning is to be used
            :return: tensor, output of upsampling and concatenation
            """
            with tf.name_scope('up_block'):
                u = up(ch, x)
                cond = concat_cond(u, with_conditioning)
                return conv(cond.shape.as_list()[-1], cond)

        def noise_to_image(sz, ch, rand_tensor, with_conditioning):
            """
            Processes random noise with fully connected layer and then convolutional layer
            :param sz: image size
            :param ch: int, number of output channels
            :param rand_tensor: tensor, input random noise to generate image
            :param with_conditioning: boolean, True if conditioning is to be used
            :return: tensor, output of convolutional layer
            """
            batch_size = rand_tensor.shape.as_list()[0]
            output_shape = [batch_size] + sz + [ch]
            with tf.name_scope('noise_to_image'):
                g_no_0 = np.prod(sz) * ch
                fc_layer = FullyConnectedLayer(
                    n_output_chns=g_no_0,
                    feature_normalization=None,
                    with_bias=True,
                    w_initializer=self.initializers['w'],
                    b_initializer=self.initializers['b'])
                g_h1p = fc_layer(rand_tensor, keep_prob=keep_prob_ph)
                g_h1p = tf.reshape(g_h1p, output_shape)
                g_h1p = concat_cond(g_h1p, with_conditioning)
                return conv(ch + conditioning_channels, g_h1p)

        def final_image(n_chns, x):
            """

            :param n_chns: int, number of output channels
            :param x: tensor, input tensor to layers
            :return: tensor, generated image
            """
            with tf.name_scope('final_image'):
                if add_noise > 0:
                    feature_shape = x.shape.as_list()[0:-1]
                    noise_shape = feature_shape + [add_noise]
                    noise = tf.random_normal(noise_shape, 0, .1)
                    x = tf.concat([x, noise], axis=3)
                conv_layer = ConvolutionalLayer(
                    n_output_chns=n_chns,
                    kernel_size=3,
                    acti_func='tanh',
                    feature_normalization=None,
                    with_bias=True,
                    w_initializer=self.initializers['w'],
                    b_initializer=self.initializers['b'])
                x_sample = conv_layer(
                    x, is_training=is_training, keep_prob=keep_prob_ph)
                return tf.image.resize_images(x_sample, image_size[:-1])

        # let the tensors flow...
        flow = random_source
        for (idx, chns) in enumerate(ch):
            if idx == 0:  # first layer fully-connected
                flow = noise_to_image(
                    sz, chns, flow, self.with_conditionings[idx])
            elif idx == len(ch) - 1:  # final conv without bn
                return final_image(chns, flow)
            else:  # upsampling block
                flow = up_block(chns, flow, self.with_conditionings[idx])


class ImageDiscriminator(BaseDiscriminator):
    """
    ### Description
        implementation of discrimator from
            Hu et al., "Freehand Ultrasound Image Simulation with Spatially-Conditioned
            Generative Adversarial Networks", MICCAI RAMBO 2017
            https://arxiv.org/abs/1707.05392

    ### Building blocks
    [FEATURE BLOCK]     - Convolutional layer (kernel size = 5, activation = selu)
                          + residual convolutiona layer (kernel size = 3, activation = selu, batch norm)
                          + convolutional layer (kernel size = 3, activation = selu, batch norm)

    [DOWN BLOCK]        - Downsampling block with residual connections:
                          Downsampling convolutional layer (stride = 2, kernel size = 3,
                          activation = selu, batch norm)
                          + residual convolutiona layer (kernel size = 3, activation = selu, batch norm)
                          + convolutional layer (kernel size = 3, activation = selu, batch norm)

    [FC]                - Fully connected layer

    If conditioning is used, it gets concatenated to the image at the discriminator input

    ### Diagram

    INPUT IMAGE --> [FEATURE BLOCK] --> [DOWN BLOCK]x5 --> [FC] --> OUTPUT LOGITS

    ### Constraints
    """
    def __init__(self, name):
        """

        :param name: layer name
        """
        super(ImageDiscriminator, self).__init__(name=name)

        w_init = tf.random_normal_initializer(0, 0.02)
        b_init = tf.constant_initializer(0.001)
        # w_init = tf.contrib.layers.variance_scaling_initializer()
        # b_init = tf.constant_initializer(0)

        self.initializers = {'w': w_init, 'b': b_init}
        self.chns = [32, 64, 128, 256, 512, 1024, 1]

    def layer_op(self, image, conditioning, is_training):
        """

        :param image: tensor, input to the network
        :param conditioning: tensor, conditioning information (e.g. coordinates in physical space)
        :param is_training: boolean, True if network is in training mode
        :return: tensor, classification logits
        """

        batch_size = image.shape.as_list()[0]

        def down(ch, x):
            """
            Downsampling convolutional layer (stride 2, kernel size 3, activation selu, batch norm)
            :param ch: int, number of output channels
            :param x: tensor, input to the convolutional layer
            :return: tensor, output of the convolutional layer
            """
            with tf.name_scope('downsample'):
                conv_layer = ConvolutionalLayer(
                    n_output_chns=ch,
                    kernel_size=3,
                    stride=2,
                    feature_normalization='batch',
                    acti_func='selu',
                    w_initializer=self.initializers['w'])
                return conv_layer(x, is_training=is_training)

        def convr(ch, x):
            """
            Convolutional layer for residuals (stride 1, kernel size 3, activation selu, batch norm)
            :param ch: int, number of output channels
            :param x: tensor, input to the convolutional layer
            :return: tensor, output of the convolutional layer
            """
            conv_layer = ConvolutionalLayer(
                n_output_chns=ch,
                kernel_size=3,
                feature_normalization='batch',
                acti_func='selu',
                w_initializer=self.initializers['w'])
            return conv_layer(x, is_training=is_training)

        def conv(ch, x, s):
            """
            Convolutional layer (stride 1, kernel size 3, batch norm)
            combining two flows
            :param ch: int, number of output channels
            :param x: tensor, input to the convolutional layer
            :param s: flow to be added after convolution
            :return: tensor, output of selu activation layer (selu(conv(x) + s))
            """
            conv_layer = ConvolutionalLayer(
                n_output_chns=ch,
                kernel_size=3,
                feature_normalization='batch',
                w_initializer=self.initializers['w'])
            acti_layer = ActiLayer(func='selu')

            # combining two flows
            res_flow = conv_layer(x, is_training=is_training) + s
            return acti_layer(res_flow)

        def down_block(ch, x):
            """
            Downsampling block with residual connections
            :param ch: int, number of output channels
            :param x: tensor, input to the convolutional layer
            :return: tensor, output of downsampling + conv + conv
            """
            with tf.name_scope('down_resnet'):
                s = down(ch, x)
                r = convr(ch, s)
                return conv(ch, r, s)

        def feature_block(ch, image):
            """
            First discriminator processing block
            :param ch: int, number of output channels
            :param image: tensor, input image to discriminator
            :return: tensor, output of conv(image) --> conv --> conv
            """
            with tf.name_scope('feature'):
                conv_layer = ConvolutionalLayer(
                    n_output_chns=ch,
                    kernel_size=5,
                    with_bias=True,
                    feature_normalization=None,
                    acti_func='selu',
                    w_initializer=self.initializers['w'],
                    b_initializer=self.initializers['b'])
                d_h1s = conv_layer(image, is_training=is_training)
                d_h1r = convr(ch, d_h1s)
                return conv(ch, d_h1r, d_h1s)

        def fully_connected(ch, features):
            """
            Final discriminator processing block
            :param ch: int, number of output channels
            :param features: tensor, input features for final classification
            :return: tensor, output logits of discriminator classification
            """
            with tf.name_scope('fully_connected'):
                # with bn?
                fc_layer = FullyConnectedLayer(
                    n_output_chns=ch, feature_normalization=None, with_bias=True)
                return fc_layer(features, is_training=is_training)

        if conditioning is not None:
            image = tf.concat([image, conditioning], axis=-1)

        # let the tensors flow...
        flow = image
        for (idx, n_chns) in enumerate(self.chns):
            if idx == 0:  # first layer
                flow = feature_block(n_chns, flow)
            elif idx == len(self.chns) - 1:  # last layer
                return fully_connected(n_chns, flow)
            else:
                flow = down_block(n_chns, flow)
