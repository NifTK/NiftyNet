# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.network.base_net import BaseNet
from niftynet.utilities.misc_common import look_up_operations
from niftynet.layer.base_layer import LayerFromCallable
from niftynet.engine import logging
from niftynet.layer.gan_blocks import GANImageBlock, BaseGenerator, BaseDiscriminator



class SimulatorGAN(GANImageBlock):
    def __init__(self, name='simulator_GAN'):
        generator = ImageGenerator(name='generator')
        discriminator = ImageDiscriminator(name='discriminator')
        super(SimulatorGAN, self).__init__(generator, discriminator, name)


class ImageGenerator(BaseGenerator):
    def __init__(self, name):
        super(ImageGenerator, self).__init__(name=name)
        self.initializers = {'w': tf.contrib.layers.variance_scaling_initializer(), 'b': tf.constant_initializer(0)}

    def layer_op(self, random_source, image_size, conditioning, is_training):
        spatial_rank = len(image_size)-1
        conditioning_channels = conditioning.get_shape().as_list()[-1] if not conditioning is None else 0
        batch_size = random_source.get_shape().as_list()[0]
        noise_size = random_source.get_shape().as_list()[1]
        generator_shortcuts = [True, True, True, True, False]

        w_init = tf.random_normal_initializer(0, 0.08)
        b_init = tf.constant_initializer(0.001)
        ch = [512]
        sz = [image_size[:-1]]
        keep_prob_ph = 1  # not passed in as a placeholder
        for i in range(4):
            ch.append(round((ch[-1]+conditioning_channels*generator_shortcuts[i+1])/2))
            sz = [[i//2 for i in sz[0]]]+sz
        if spatial_rank == 3:
            def resize_func(x, sz):
                sz_x = x.get_shape().as_list()
                r1 = tf.image.resize_images(tf.reshape(x, sz_x[:3]+[-1]), sz[0:2])
                r2 = tf.image.resize_images(tf.reshape(r1, [sz_x[0], sz[0]*sz[1], sz_x[3], -1]), [sz[0]*sz[1], sz[2]])
                return tf.reshape(r2, [sz_x[0]]+sz+[sz_x[-1]])
        elif spatial_rank == 2:
                resize_func = tf.image.resize_bilinear
            
        def concat_cond(x, i):
            if not conditioning is None and generator_shortcuts[i]:
                with tf.name_scope('concat_conditioning'):
                    return tf.concat([x, resize_func(conditioning, x.get_shape().as_list()[1:-1])], axis=3)
            else:
                return x
            
        def conv(ch, x):
            with tf.name_scope('conv'):
                return tf.nn.relu(ConvolutionalLayer(ch, 3, w_initializer=w_init)(x, is_training=is_training))
            
        def up(ch, x):
            with tf.name_scope('up'):
                deconv=DeconvolutionalLayer(ch, 3, stride=2, w_initializer=w_init)(x, is_training=is_training)
                # deconv=tf.nn.conv2d(tf.image.resize_bilinear(g_h1, data_size_3), G_W2t, strides_none, "SAME")
                return tf.nn.relu(deconv)

        def up_block(ch, x, i):
            with tf.name_scope('up_block'):
                cond=concat_cond(up(ch, x), i)
                return conv(cond.get_shape().as_list()[-1], cond)
        with tf.name_scope('noise_to_image'):
            g_no_0 = np.prod(sz[0])*ch[0]
            w1p = tf.get_variable("G_W1p", shape=[noise_size, g_no_0], initializer=w_init)
            b1p = tf.get_variable('G_b1p', shape=[g_no_0], initializer=b_init)
            g_h1p = tf.nn.dropout(tf.nn.relu(tf.matmul(random_source, w1p) + b1p), keep_prob_ph)
            g_h1p = tf.reshape(g_h1p, [batch_size]+sz[0]+[ch[0]])
            g_h1p = concat_cond(g_h1p, 0)
            g_h1 = conv(ch[0]+conditioning_channels, g_h1p)
        g_h2 = up_block(ch[1], g_h1, 1)
        g_h3 = up_block(ch[2], g_h2, 2)
        g_h4 = up_block(ch[3], g_h3, 3)
        g_h5 = up_block(ch[4], g_h4, 4)  # did not implement different epsilon
        with tf.name_scope('final_image'):
            x_sample = ConvolutionalLayer(1, 3, with_bn=False, w_initializer=w_init)(g_h5, is_training=is_training)
            x_sample = tf.nn.dropout(2.*tf.nn.tanh(x_sample), keep_prob_ph)
        with tf.name_scope('summaries_verbose'):
            tf.summary.histogram('hist_g_h2',g_h2,[logging.LOG])
            tf.summary.histogram('hist_g_h3',g_h3,[logging.LOG])
            tf.summary.histogram('hist_g_h4',g_h4,[logging.LOG])
            tf.summary.histogram('hist_g_h5',g_h5,[logging.LOG])
            tf.summary.histogram('hist_img',x_sample,[logging.LOG])
            [tf.summary.histogram(v.name,v,[logging.LOG]) for v in self.trainable_variables()]
        return x_sample


class ImageDiscriminator(BaseDiscriminator):
    def __init__(self, name):
        super(ImageDiscriminator, self).__init__(name=name)
        self.initializers = {'w': tf.contrib.layers.variance_scaling_initializer(), 'b': tf.constant_initializer(0)}

    def layer_op(self, image, conditioning, is_training):
        batch_size = image.get_shape().as_list()[0]

        w_init = tf.random_normal_initializer(0, 0.2)
        b_init = tf.constant_initializer(0.001)

        def leaky_relu(x, alpha=0.2):
            with tf.name_scope('leaky_relu'):
                return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)
        ch = [32, 64, 128, 256, 512, 1024]

        def down(ch, x):
            with tf.name_scope('downsample'):
                c = ConvolutionalLayer(ch, 3, stride=2, w_initializer=w_init)(x, is_training=is_training)
                return leaky_relu(c)

        def convr(ch, x):
            return leaky_relu(ConvolutionalLayer(ch, 3, w_initializer=w_init)(x, is_training=is_training))

        def conv(ch, x, s): 
            return leaky_relu(ConvolutionalLayer(ch, 3, w_initializer=w_init)(x, is_training=is_training)+s)

        def down_block(ch, x):
            with tf.name_scope('down_resnet'):
                s = down(ch, x)
                r = convr(ch, s)
                return conv(ch, r, s)

        if not conditioning is None:
            image = tf.concat([image, conditioning], axis=-1)
        with tf.name_scope('feature'):
            d_h1s = leaky_relu(ConvolutionalLayer(ch[0], 5, with_bias=True,
                                                  with_bn=False,
                                                  w_initializer=w_init)(image,
                                                                        is_training=is_training))
            d_h1r = convr(ch[0], d_h1s)
            d_h1 = conv(ch[0], d_h1r, d_h1s)
        d_h2 = down_block(ch[1], d_h1)
        d_h3 = down_block(ch[2], d_h2)
        d_h4 = down_block(ch[3], d_h3)
        d_h5 = down_block(ch[4], d_h4)
        d_h6 = down_block(ch[5], d_h5)
        with tf.name_scope('fc'):
            d_hf = tf.reshape(d_h6, [batch_size, -1])
            d_nf_o = np.prod(d_hf.get_shape().as_list()[1:])
            D_Wo = tf.get_variable("D_Wo", shape=[d_nf_o, 1], initializer=w_init)/d_nf_o
            D_bo = tf.get_variable('D_bo', shape=[1], initializer=b_init)

            d_logit = tf.matmul(d_hf, D_Wo) + D_bo
        with tf.name_scope('summaries_verbose'):
            tf.summary.histogram('hist_d_h2',d_h2,[logging.LOG])
            tf.summary.histogram('hist_d_h3',d_h3,[logging.LOG])
            tf.summary.histogram('hist_d_h4',d_h4,[logging.LOG])
            tf.summary.histogram('hist_d_h5',d_h5,[logging.LOG])
            tf.summary.histogram('hist_d_h6',d_h6,[logging.LOG])
            tf.summary.histogram('hist_d_logit',d_logit,[logging.LOG])
            [tf.summary.histogram(v.name,v,[logging.LOG]) for v in self.trainable_variables()]
        return d_logit
