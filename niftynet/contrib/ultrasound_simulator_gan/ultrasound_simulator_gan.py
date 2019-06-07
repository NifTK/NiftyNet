# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.gan_blocks import GANImageBlock, BaseGenerator, BaseDiscriminator


class SimulatorGAN(GANImageBlock):
    def __init__(self, load_from_checkpoint=True, name='simulator_GAN'):
        generator = ImageGenerator(load_from_checkpoint=load_from_checkpoint, name='generator')
        discriminator = ImageDiscriminator(load_from_checkpoint=load_from_checkpoint, name='discriminator')
        super(SimulatorGAN, self).__init__(generator, discriminator, clip=None, name=name)


class ImageGenerator(BaseGenerator):
    def __init__(self, load_from_checkpoint=True, name='generator'):
        self.load_from_checkpoint = load_from_checkpoint
        super(ImageGenerator, self).__init__(name=name)
        self.initializers = {'w': tf.contrib.layers.variance_scaling_initializer(), 'b': tf.constant_initializer(0)}
        self.noise_channels_per_layer = 0
        self.generator_shortcuts = [True, True, True, True, False]

    def layer_op(self, random_source, image_size, conditioning, is_training):
        spatial_rank = len(image_size) - 1
        add_noise = self.noise_channels_per_layer
        conditioning_channels = conditioning.shape.as_list()[
                                    -1] + add_noise if not conditioning is None else add_noise
        batch_size = random_source.shape.as_list()[0]
        noise_size = random_source.shape.as_list()[1]

        w_init = tf.random_normal_initializer(0, 0.02)
        b_init = tf.constant_initializer(0.001)
        ch = [512]
        sz = [[160, 120]]

        keep_prob_ph = 1  # not passed in as a placeholder
        for i in range(4):
            ch.append(round((ch[-1] + conditioning_channels * self.generator_shortcuts[i]) / 2))
            sz = [[round(i / 2) for i in sz[0]]] + sz
        if spatial_rank == 3:
            def resize_func(x, sz):
                sz_x = x.shape.as_list()
                r1 = tf.image.resize_images(tf.reshape(x, sz_x[:3] + [-1]), sz[0:2])
                r2 = tf.image.resize_images(tf.reshape(r1, [sz_x[0], sz[0] * sz[1], sz_x[3], -1]),
                                            [sz[0] * sz[1], sz[2]])
                return tf.reshape(r2, [sz_x[0]] + sz + [sz_x[-1]])
        elif spatial_rank == 2:
            resize_func = tf.image.resize_bilinear

        def concat_cond(x, i):
            if add_noise:
                noise = [tf.random_normal(x.shape.as_list()[0:-1] + [add_noise], 0, .1)]
            else:
                noise = []
            if not conditioning is None and self.generator_shortcuts[i]:
                with tf.name_scope('concat_conditioning'):
                    return tf.concat([x, resize_func(conditioning, x.shape.as_list()[1:-1])] + noise, axis=3)
            else:
                return x

        def conv(ch, x):
            with tf.name_scope('conv'):
                conv_layer = ConvolutionalLayer(ch, 3, feature_normalization=None, w_initializer=w_init)
                c = conv_layer(x, is_training=is_training)
                return tf.nn.relu(tf.contrib.layers.batch_norm(c))

        def up(ch, x, hack=False):
            with tf.name_scope('up'):
                deconv = DeconvolutionalLayer(ch, 3, feature_normalization=None, stride=2, w_initializer=w_init)(x, is_training=is_training)
                if hack:
                    deconv = deconv[:, :, 1:, :]  # hack to match Yipeng's image size
                return tf.nn.relu(tf.contrib.layers.batch_norm(deconv))

        def up_block(ch, x, i, hack=False):
            with tf.name_scope('up_block'):
                cond = concat_cond(up(ch, x, hack), i)
                return conv(cond.shape.as_list()[-1], cond)

        with tf.name_scope('noise_to_image'):
            g_no_0 = np.prod(sz[0]) * ch[0]
            w1p = tf.get_variable("G_W1p", shape=[noise_size, g_no_0], initializer=w_init)
            b1p = tf.get_variable('G_b1p', shape=[g_no_0], initializer=b_init)
            g_h1p = tf.nn.dropout(tf.nn.relu(tf.matmul(random_source, w1p) + b1p), keep_prob_ph)
            g_h1p = tf.reshape(g_h1p, [batch_size] + sz[0] + [ch[0]])
            g_h1p = concat_cond(g_h1p, 0)
            g_h1 = conv(ch[0] + conditioning_channels, g_h1p)
        g_h2 = up_block(ch[1], g_h1, 1, hack=True)
        g_h3 = up_block(ch[2], g_h2, 2)
        g_h4 = up_block(ch[3], g_h3, 3)
        g_h5 = up_block(ch[4], g_h4, 4)
        with tf.name_scope('final_image'):
            if add_noise:
                noise = tf.random_normal(g_h5.shape.as_list()[0:-1] + [add_noise], 0, .1)
                g_h5 = tf.concat([g_h5, noise],axis=3)
            x_sample = ConvolutionalLayer(1, 3, feature_normalization=None, with_bias=True,
                                          w_initializer=w_init,
                                          b_initializer=b_init)(g_h5, is_training=is_training)
            x_sample = tf.nn.dropout(tf.nn.tanh(x_sample), keep_prob_ph)
        if self.load_from_checkpoint:
            checkpoint_name = '/home/egibson/deeplearning/TensorFlow/NiftyNet/NiftyNet/' + \
                              'contrib/ultrasound_simulator_gan/ultrasound_simulator_gan'
            restores = [
                ['G_Wo', 'generator/conv_5/conv_/w'],
                ['G_W5t', 'generator/deconv_3/deconv_/w'],
                ['G_W5', 'generator/conv_4/conv_/w'],
                ['G_W4t', 'generator/deconv_2/deconv_/w'],
                ['G_W4', 'generator/conv_3/conv_/w'],
                ['G_W3t', 'generator/deconv_1/deconv_/w'],
                ['G_W3', 'generator/conv_2/conv_/w'],
                ['G_W2t', 'generator/deconv/deconv_/w'],
                ['G_W2', 'generator/conv_1/conv_/w'],
                ['G_W1p', 'generator/G_W1p'],
                ['G_W1', 'generator/conv/conv_/w'],
                ['G_b1p', 'generator/G_b1p']]
            [tf.add_to_collection('NiftyNetObjectsToRestore', (r[1], checkpoint_name, r[0]))
             for r in restores]
        return x_sample


class ImageDiscriminator(BaseDiscriminator):
    def __init__(self, load_from_checkpoint=True, name='discriminator'):
        self.load_from_checkpoint=load_from_checkpoint
        super(ImageDiscriminator, self).__init__(name=name)
        self.initializers = {'w': tf.contrib.layers.variance_scaling_initializer(),
                             'b': tf.constant_initializer(0)}

    def layer_op(self, image, conditioning, is_training):
        conditioning = tf.image.resize_images(conditioning, [160, 120])

        batch_size = image.shape.as_list()[0]

        w_init = tf.random_normal_initializer(0, 0.02)
        b_init = tf.constant_initializer(0.001)

        def leaky_relu(x, alpha=0.2):
            with tf.name_scope('leaky_relu'):
                return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

        ch = [32, 64, 128, 256, 512, 1024]

        def down(ch, x):
            with tf.name_scope('downsample'):
                c = ConvolutionalLayer(ch, 3, stride=2, feature_normalization=None,
                                       w_initializer=w_init)(x, is_training=is_training)
                c=tf.contrib.layers.batch_norm(c)
                c = leaky_relu(c)
                return c

        def convr(ch, x):
            c= ConvolutionalLayer(ch, 3, feature_normalization=None,
                                  w_initializer=w_init)(x, is_training=is_training)
            return leaky_relu(tf.contrib.layers.batch_norm(c))
        def conv(ch, x, s):
            c = (ConvolutionalLayer(ch, 3, feature_normalization=None,
                                      w_initializer=w_init)(x, is_training=is_training))
            return leaky_relu(tf.contrib.layers.batch_norm(c) + s)

        def down_block(ch, x):
            with tf.name_scope('down_resnet'):
                s = down(ch, x)
                r = convr(ch, s)
                return conv(ch, r, s)

        if not conditioning is None:
            image = tf.concat([image, conditioning], axis=-1)
        with tf.name_scope('feature'):
            d_h1s = ConvolutionalLayer(ch[0], 5, with_bias=True,
                                                  feature_normalization=None,
                                                  w_initializer=w_init,
                                                  b_initializer=b_init)(image,
                                                                        is_training=is_training)

            d_h1s = leaky_relu(d_h1s)
            d_h1r = convr(ch[0], d_h1s)
            d_h1 = conv(ch[0], d_h1r, d_h1s)
        d_h2 = down_block(ch[1], d_h1)
        d_h3 = down_block(ch[2], d_h2)
        d_h4 = down_block(ch[3], d_h3)
        d_h5 = down_block(ch[4], d_h4)
        d_h6 = down_block(ch[5], d_h5)
        with tf.name_scope('fc'):
            d_hf = tf.reshape(d_h6, [batch_size, -1])
            d_nf_o = np.prod(d_hf.shape.as_list()[1:])
            D_Wo = tf.get_variable("D_Wo", shape=[d_nf_o, 1], initializer=w_init)
            D_bo = tf.get_variable('D_bo', shape=[1], initializer=b_init)

            d_logit = tf.matmul(d_hf, D_Wo) + D_bo
        if self.load_from_checkpoint:
            checkpoint_name = '/home/egibson/deeplearning/TensorFlow/NiftyNet/NiftyNet/' + \
                              'contrib/ultrasound_simulator_gan/ultrasound_simulator_gan'
            restores = [
                ['D_b1s', 'discriminator/conv/conv_/b'],
                ['D_W1s', 'discriminator/conv/conv_/w'],
                ['D_W1r1', 'discriminator/conv_1/conv_/w'],
                ['D_W1r2', 'discriminator/conv_2/conv_/w'],
                ['D_W2s', 'discriminator/conv_3/conv_/w'],
                ['D_W2r1', 'discriminator/conv_4/conv_/w'],
                ['D_W2r2', 'discriminator/conv_5/conv_/w'],
                ['D_W3s', 'discriminator/conv_6/conv_/w'],
                ['D_W3r1', 'discriminator/conv_7/conv_/w'],
                ['D_W3r2', 'discriminator/conv_8/conv_/w'],
                ['D_W4s', 'discriminator/conv_9/conv_/w'],
                ['D_W4r1', 'discriminator/conv_10/conv_/w'],
                ['D_W4r2', 'discriminator/conv_11/conv_/w'],
                ['D_W5s', 'discriminator/conv_12/conv_/w'],
                ['D_W5r1', 'discriminator/conv_13/conv_/w'],
                ['D_W5r2', 'discriminator/conv_14/conv_/w'],
                ['D_W6s', 'discriminator/conv_15/conv_/w'],
                ['D_W6r1', 'discriminator/conv_16/conv_/w'],
                ['D_W6r2', 'discriminator/conv_17/conv_/w'],
                ['D_Wo', 'discriminator/D_Wo'],
                ['D_bo', 'discriminator/D_bo']]
            [tf.add_to_collection('NiftyNetObjectsToRestore', (r[1], checkpoint_name, r[0]))
             for r in restores]
        return d_logit
