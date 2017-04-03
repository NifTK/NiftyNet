# -*- coding: utf-8 -*-
import tensorflow as tf
from base_layer import BaseLayer
from network.net_template import NetTemplate

# implementation of V-Net:
#   Milletari et al., "V-Net: Fully convolutional neural networks for
#   volumetric medical image segmentation", 3DV 2016

class VNet(NetTemplate):
    def __init__(self,
                 batch_size,
                 image_size,
                 label_size,
                 num_classes,
                 is_training=True,
                 device_str="cpu"):
        super(VNet, self).__init__(batch_size,
                                   image_size,
                                   label_size,
                                   num_classes,
                                   is_training,
                                   device_str)
        assert(image_size % 8 == 0) # image_size should be divisible by 8
        self.num_fea = [16, 32, 64, 128, 256]
        self.set_activation_type('prelu')
        self.name = "VNet"
        print "{}\n"\
            "3x3x3 convolution {} kernels\n" \
            "Classifiying {} classes".format(
                self.name, self.num_fea, self.num_classes)


    def inference(self, images, layer_id=None):
        BaseLayer._print_activations(images)
        print ""
        images = tf.expand_dims(images, 4)
        pad_images = tf.tile(images, [1, 1, 1, 1, self.num_fea[0]])
        with tf.variable_scope('L1') as scope:
            res_1, down_1 = self._res_block_5x5(
                images, pad_images, 1, self.num_fea[0], self.num_fea[1],
                'downsample')

        with tf.variable_scope('L2') as scope:
            res_2, down_2 = self._res_block_5x5(
                down_1, down_1, 2, self.num_fea[1], self.num_fea[2],
                'downsample')

        with tf.variable_scope('L3') as scope:
            res_3, down_3 = self._res_block_5x5(
                down_2, down_2, 3, self.num_fea[2], self.num_fea[3],
                'downsample')

        with tf.variable_scope('L4') as scope:
            res_4, down_4 = self._res_block_5x5(
                down_3, down_3, 3, self.num_fea[3], self.num_fea[4],
                'downsample')

        with tf.variable_scope('V') as scope:
            _, up_4 = self._res_block_5x5(
                down_4, down_4, 3, self.num_fea[4], self.num_fea[4],
                'upsample')

        with tf.variable_scope('R4') as scope:
            concat_r4 = tf.concat([up_4, res_4], 4)
            _, up_3 = self._res_block_5x5(
                concat_r4, up_4, 3, self.num_fea[4], self.num_fea[3],
                'upsample')

        with tf.variable_scope('R3') as scope:
            concat_r3 = tf.concat([up_3, res_3], 4)
            _, up_2 = self._res_block_5x5(
                concat_r3, up_3, 3, self.num_fea[3], self.num_fea[2],
                'upsample')

        with tf.variable_scope('R2') as scope:
            concat_r2 = tf.concat([up_2, res_2], 4)
            _, up_1 = self._res_block_5x5(
                concat_r2, up_2, 2, self.num_fea[2], self.num_fea[1],
                'upsample')

        with tf.variable_scope('R1') as scope:
            concat_r1 = tf.concat([up_1, res_1], 4)
            _, conv_fc = self._res_block_5x5(
                concat_r1, up_1, 1, self.num_fea[1], self.num_classes,
                'conv_1x1x1')

        if layer_id is None:
            return conv_fc

    def _res_block_5x5(self, f_in, res_in,
                       n_blocks, n_middle, n_out,
                       type_str='downsample'):
        # the residual block
        ni_ = f_in.get_shape()[-1].value
        for i in range(n_blocks):
            with tf.variable_scope('5x5_conv_%d'%i) as scope:
                conv = self.conv_5x5(f_in, ni_, n_middle)
                f_in = self.nonlinear_acti(conv) if i < n_blocks - 1 else conv
                ni_ = n_middle
        res_f = res_in + f_in
        # the main branch
        if type_str is 'downsample':
            conv_f_in = self.downsample_conv_2x2(res_f, n_middle, n_out)
        elif type_str is 'upsample':
            conv_f_in = self.upsample_conv_2x2(res_f, n_middle, n_out)
        elif type_str is 'conv_1x1x1':
            conv_f_in = self.conv_layer_1x1(res_f, n_middle, n_out,
                                            bn=False, acti=False)
        conv_f_in = self.nonlinear_acti(conv_f_in)
        BaseLayer._print_activations(conv_f_in)
        print ""
        return res_f, conv_f_in
