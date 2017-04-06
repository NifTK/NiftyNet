# -*- coding: utf-8 -*-
import tensorflow as tf
from six.moves import range

from network.base_layer import BaseLayer
from network.net_template import NetTemplate


# implementation of HighRes3DNet:
#    Li et al., "On the compactness, efficiency, and representation of 3D
#    convolutional networks: Brain parcellation as a pretext task", IPMI '17
class HighRes3DNet(NetTemplate):
    def __init__(self,
                 batch_size,
                 image_size,
                 label_size,
                 num_classes,
                 is_training=True,
                 device_str="cpu"):
        super(HighRes3DNet, self).__init__(batch_size,
                                           image_size,
                                           label_size,
                                           num_classes,
                                           is_training,
                                           device_str)
        assert(image_size % 4 == 0)
        self.num_res_blocks = [3, 3, 3]
        self.num_features = [16, 32, 64, 80]
        self.set_activation_type('relu')
        #self.num_features = [16, 32, 48, 48] # prelu increases the model size
        #self.set_activation_type('prelu')
        self.name = "HighRes3DNet\n"\
            "{} dilat-0 blocks with {} features\n"\
            "{} dilat-2 blocks with {} features\n"\
            "{} dilat-4 blocks with {} features\n"\
            "{} FC features to classify {} classes".format(
                self.num_res_blocks[0], self.num_features[0],
                self.num_res_blocks[1], self.num_features[1],
                self.num_res_blocks[2], self.num_features[2],
                self.num_features[3], num_classes)
        print 'using {}'.format(self.name)

    def inference(self, images, layer_id=None):
        BaseLayer._print_activations(images)
        zero_paddings = [[0, 0], [0, 0], [0, 0]]
        with tf.variable_scope('conv_1_1') as scope:
            conv_1_1 = self.conv_3x3(images, 1, self.num_features[0])
            conv_1_1 = self.batch_norm(conv_1_1)
            conv_1_1 = self.nonlinear_acti(conv_1_1)
            BaseLayer._print_activations(conv_1_1)

        with tf.variable_scope('res_1') as scope:
            res_1 = self._res_block(conv_1_1,
                                    self.num_features[0],
                                    self.num_features[0],
                                    self.num_res_blocks[0])

        ## convolutions  dilation factor = 2
        with tf.variable_scope('dilate_1_start') as scope:
            res_1 = tf.space_to_batch_nd(res_1, [2, 2, 2], zero_paddings)
            BaseLayer._print_activations(res_1)
        with tf.variable_scope('res_2') as scope:
            res_2 = self._res_block(res_1,
                                    self.num_features[0],
                                    self.num_features[1],
                                    self.num_res_blocks[1])
        with tf.variable_scope('dilate_1_end') as scope:
            res_2 = tf.batch_to_space_nd(res_2, [2, 2, 2], zero_paddings)
            BaseLayer._print_activations(res_2)

        ## convolutions  dilation factor = 4
        with tf.variable_scope('dilate_2_start') as scope:
            res_2 = tf.space_to_batch_nd(res_2, [4, 4, 4], zero_paddings)
            BaseLayer._print_activations(res_2)
        with tf.variable_scope('res_3') as scope:
            res_3 = self._res_block(res_2,
                                    self.num_features[1],
                                    self.num_features[2],
                                    self.num_res_blocks[2])
        with tf.variable_scope('dilate_2_end') as scope:
            res_3 = tf.batch_to_space_nd(res_3, [4, 4, 4], zero_paddings)
            BaseLayer._print_activations(res_3)

        ## 1x1x1 convolution "fully connected"
        with tf.variable_scope('conv_fc_1') as scope:
            conv_fc = self.conv_layer_1x1(res_3,
                                          self.num_features[2],
                                          self.num_features[3],
                                          bias=True, bn=True, acti=True)
            BaseLayer._print_activations(conv_fc)

        with tf.variable_scope('conv_fc_2') as scope:
            conv_fc = self.conv_layer_1x1(conv_fc,
                                          self.num_features[3],
                                          self.num_classes,
                                          bias=True, bn=True, acti=False)
            BaseLayer._print_activations(conv_fc)

        if layer_id == 'conv_features':
            return res_3

        if layer_id is None:
            return conv_fc

    def _res_block(self, f_in, ni_, no_,
                   n_blocks, conv_type=("3x3", "3x3")):
        if n_blocks == 0:
            return f_in
        for b in range(n_blocks):
            with tf.variable_scope('block_%d' % b) as scope:
                conv_fea = self._multiple_conv(f_in, ni_, no_, conv_type)
                res_out = self._res_connect(f_in, conv_fea)
            f_in = res_out
            ni_ = no_
        BaseLayer._print_activations(res_out)
        print('//repeated {:d} times'.format(n_blocks * len(conv_type)))
        return f_in

    def _multiple_conv(self, f_in, ni_, no_, conv_type=("3x3", "3x3")):
        conv = []
        for t in conv_type:
            if t == "3x3":
                conv.append(self.conv_3x3)
            elif t == "1x1":
                conv.append(self.conv_1x1)
            else:
                raise ValueError('Conv type %s not supported' % t)
        for i, conv_layer in enumerate(conv):
            with tf.variable_scope('conv_%d' % i) as scope:
                f_out = self.batch_norm(f_in)
                f_out = self.nonlinear_acti(f_out)
                f_out = conv_layer(f_out, ni_, no_)
            f_in = f_out
            ni_ = no_
        return f_out

    def _res_connect(self, f_in, conv_in):
        # return element-wise sum of f_in and conv_in (handles dim mis-match)
        n_in = f_in.get_shape()[-1].value
        n_conv = conv_in.get_shape()[-1].value
        with tf.variable_scope('shortcut') as scope:
            if n_in == n_conv:
                return f_in + conv_in
            elif n_in < n_conv:  # pad the channel dim
                pad_1 = (n_conv - n_in) // 2
                pad_2 = n_conv - n_in - pad_1
                padded_f_in = tf.pad(
                    f_in, [[0, 0], [0, 0], [0, 0], [0, 0], [pad_1, pad_2]])
                return conv_in + padded_f_in
            elif n_in > n_conv:  # make a projection
                proj_f_in = self.conv_1x1(f_in, n_in, n_conv)
                return conv_in + proj_f_in
        return
