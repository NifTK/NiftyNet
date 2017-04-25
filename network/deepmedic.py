# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from network.base_layer import BaseLayer
from network.net_template import NetTemplate

"""
reimplementation of DeepMedic:
  Kamnitsas et al., "Efficient multi-scale 3D CNN with fully connected
  CRF for accurate brain lesion segmentation", MedIA '17
"""
class DeepMedic(NetTemplate):
    def __init__(self,
                 batch_size,
                 image_size,
                 label_size,
                 num_classes,
                 is_training=True,
                 device_str="cpu"):
        super(DeepMedic, self).__init__(batch_size,
                                     image_size,
                                     label_size,
                                     num_classes,
                                     is_training,
                                     device_str)
        # image_size is defined as the largest context, then:
        #   downsampled path size: image_size / d_factor
        #   downsampled path output: image_size / d_factor - 16

        # to make sure same size of feature maps from both pathways:
        #   normal path size: (image_size / d_factor - 16) * d_factor + 16
        #   normal path output: (image_size / d_factor - 16) * d_factor

        # where 16 is fixed by the receptive field of conv layers
        # TODO: make sure label_size = image_size/d_factor - 16

        self.d_factor = 3 # subsample factor
        self.crop_diff = (self.d_factor - 1) * 16
        assert(image_size % self.d_factor == 0)
        assert(self.d_factor % 2 == 1) # to make the downsampling centered
        assert(image_size % 2 == 1) # to make the crop centered
        assert(image_size > self.d_factor * 16) # minimum receptive field

        self.conv_fea = [30, 30, 40, 40, 40, 40, 50, 50]
        self.fc_fea = [150, 150]
        self.set_activation_type('prelu')
        self.name = "DeepMedic"
        print("{}\n"\
            "3x3x3 convolution {} kernels\n" \
            "Classifiying {} classes".format(
                self.name, self.conv_fea, self.num_classes))


    def inference(self, images, layer_id=None):
        BaseLayer._print_activations(images)
        img_1 = self._crop(images)
        img_2 = self._downsample(images)

        # two pathways for convolutional layers
        conv_1, conv_2 = self.conv_layers(img_1, img_2)
        # upsample the previously downsampled pathway
        conv_2 = self._upsample(conv_2)
        # combine both
        combined = tf.concat([conv_1, conv_2], 4)
        # "fully connnected layers"
        fc = self.fc_layers(combined)

        if layer_id is None:
            return fc

    def conv_layers(self, conv_1, conv_2):
        ni_ = 1
        for k, no_ in enumerate(self.conv_fea):
            with tf.variable_scope('pathway_1_%d' % k) as scope:
                conv_1 = self.conv_layer_3x3(conv_1, ni_, no_, padding='VALID')
            with tf.variable_scope('pathway_2_%d' % k) as scope:
                conv_2 = self.conv_layer_3x3(conv_2, ni_, no_, padding='VALID')
            ni_ = no_
            BaseLayer._print_activations(conv_2)
        return conv_1, conv_2

    def fc_layers(self, f_in):
        ni_ = self.conv_fea[-1] * 2
        for k, no_ in enumerate(self.fc_fea):
            with tf.variable_scope('conv_fc_%d' % k) as scope:
                f_in = self.conv_layer_1x1(f_in, ni_, no_)
            ni_ = no_
            BaseLayer._print_activations(f_in)
        with tf.variable_scope('conv_fc_%d' % (k+1)) as scope:
            f_in = self.conv_layer_1x1(f_in, ni_, self.num_classes)
        BaseLayer._print_activations(f_in)
        return f_in

    def _upsample(self, fea_maps):
        # upsample by a factor of self.d_factor
        fea_maps = tf.tile(fea_maps, [self.d_factor**3, 1, 1, 1, 1])
        fea_maps = tf.batch_to_space_nd(
            fea_maps, [self.d_factor, self.d_factor, self.d_factor],
            [[0, 0], [0, 0], [0, 0]])
        return fea_maps


    def _downsample(self, images):
        # downsample the larger context to get the downsampled pathway
        # by a factor of self.d_factor
        np_kernel = np.zeros((self.d_factor,
                              self.d_factor,
                              self.d_factor, 1, 1))
        center_ = np.int(self.d_factor / 2)
        np_kernel[center_, center_, center_, 0, 0] = 1
        d_kernel = tf.constant(np_kernel, dtype=tf.float32)
        downsampled = tf.nn.conv3d(
            images, d_kernel,
            [1, self.d_factor, self.d_factor, self.d_factor, 1],
            padding='VALID')
        return downsampled


    def _crop(self, images):
        # crop the normal pathway input from a larger context
        np_kernel = np.zeros((self.crop_diff + 1,
                              self.crop_diff + 1,
                              self.crop_diff + 1, 1, 1))
        center_ = np.int(self.crop_diff / 2)
        np_kernel[center_, center_, center_, 0, 0] = 1
        crop_kernel = tf.constant(np_kernel, dtype=tf.float32)
        cropped = tf.nn.conv3d(
            images, crop_kernel, [1, 1, 1, 1, 1], padding='VALID')
        return cropped
