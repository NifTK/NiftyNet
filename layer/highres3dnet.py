# -*- coding: utf-8 -*-
import tensorflow as tf
from base import Layer
from highresblock import HighResBlock
from convolution import ConvolutionalLayer
from six.moves import range


"""
implementation of HighRes3DNet:
  Li et al., "On the compactness, efficiency, and representation of 3D
  convolutional networks: Brain parcellation as a pretext task", IPMI '17
"""
class HighRes3DNet(Layer):
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_type='relu',
                 name='HighRes3DNet'):

        super(HighRes3DNet, self).__init__(name='HighRes3DNet')
        self.w_initializer = w_initializer
        self.w_regularizer = w_regularizer
        self.b_initializer = b_initializer
        self.b_regularizer = b_regularizer

        self.num_res_blocks = [3, 3, 3]
        self.num_features = [16, 32, 64, 80]
        self.num_classes = num_classes
        self.name = "HighRes3DNet\n"\
            "{} dilat-0 blocks with {} features\n"\
            "{} dilat-2 blocks with {} features\n"\
            "{} dilat-4 blocks with {} features\n"\
            "{} FC features to classify {} classes".format(
                self.num_res_blocks[0], self.num_features[0],
                self.num_res_blocks[1], self.num_features[1],
                self.num_res_blocks[2], self.num_features[2],
                self.num_features[3], num_classes)
        print('using {}'.format(self.name))

    def layer_op(self, images, is_training, layer_id=None):
        zero_paddings = [[0,0]]*3
        assert(images.get_shape()[1] % 4 == 0)
        conv_1_op = ConvolutionalLayer(self.num_features[0], kernel_size=3, stride=1, name='conv_1')
        conv_1_out = conv_1_op(images, is_training)
        res_out = conv_1_out
        for i in range(self.num_res_blocks[0]):
            res_block = HighResBlock(self.num_features[0], kernels=(3, 3), with_res=True, name='res_1_{}'.format(i))
            res_out = res_block(res_out, is_training)


        ## convolutions  dilation factor = 2
        res_out = tf.space_to_batch_nd(res_out, [2, 2, 2], zero_paddings)
        for i in range(self.num_res_blocks[1]):
            res_block = HighResBlock(self.num_features[1], kernels=(3, 3), with_res=True, name='res_2_{}'.format(i))
            res_out = res_block(res_out, is_training)
        res_out = tf.batch_to_space_nd(res_out, [2, 2, 2], zero_paddings)

        ## convolutions  dilation factor = 4
        res_out = tf.space_to_batch_nd(res_out, [4, 4, 4], zero_paddings)
        for i in range(self.num_res_blocks[2]):
            res_block = HighResBlock(self.num_features[2], kernels=(3, 3), with_res=True, name='res_3_{}'.format(i))
            res_out = res_block(res_out, is_training)
        res_out = tf.batch_to_space_nd(res_out, [4, 4, 4], zero_paddings)

        ## 1x1x1 convolution "fully connected"
        conv_kernel_1_op = ConvolutionalLayer(self.num_features[3], kernel_size=1, stride=1, padding='SAME', name='con_fc_1')
        conv_fc_out = conv_kernel_1_op(res_out, is_training)

        ## 1x1x1 convolution to num_classes
        conv_kernel_1_op = ConvolutionalLayer(self.num_classes, kernel_size=1, stride=1, padding='SAME', name='con_fc_2')
        conv_fc_out = conv_kernel_1_op(conv_fc_out, is_training)

        if layer_id == 'conv_features':
            return res_out

        if layer_id is None:
            return conv_fc_out
