# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.downsample import DownSampleLayer as Down
from niftynet.layer.activation import ActiLayer as Acti
from niftynet.layer.grid_warper import AffineGridWarperLayer as Grid
from niftynet.layer.resampler import ResamplerLayer as Resampler
from niftynet.layer.fully_connected import FullyConnectedLayer as FC
from niftynet.network.base_net import BaseNet


class INetGlobal(BaseNet):
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='inet-global'):

        BaseNet.__init__(self,
                         num_classes=num_classes,
                         w_initializer=w_initializer,
                         w_regularizer=w_regularizer,
                         b_initializer=w_initializer,
                         b_regularizer=b_regularizer,
                         name=name)
        # TODO initializer

        # self.num_channel_initial_global = 4
        # self.conv_size_initial = 3
        self.nc0_g = [4, 8, 16, 32, 64]
        self.k_conv = 3
        self.k_pool = [1, 2, 2, 1]
        self.strides_down = [1, 2, 2, 2, 1]


    def layer_op(self,
                 fixed_image, fixed_label, moving_image, moving_label,
                 is_training=True):

        images = tf.concat([moving_image, fixed_image], axis=-1)
        print(images)

        def res_block(inputs, n_chns=4):
            conv_0 = Conv(n_output_chns=n_chns,
                         kernel_size=self.k_conv,
                         acti_func=self.acti_func)(inputs, is_training)
            conv_1 = Conv(n_output_chns=n_chns,
                          kernel_size=self.k_conv,
                          acti_func=self.acti_func)(conv_0, is_training)
            conv_2 = Conv(n_output_chns=n_chns,
                          kernel_size=self.k_conv,
                          acti_func=None)(conv_1, is_training)
            conv_res = Acti(self.acti_func)(conv_2 + conv_0)
            conv_down = Down('MAX', kernel_size=2, stride=2)(conv_res)
            return conv_down

        res_1 = res_block(images, n_chns=self.nc0_g[0])
        res_2 = res_block(res_1, n_chns=self.nc0_g[1])
        res_3 = res_block(res_2, n_chns=self.nc0_g[2])
        res_4 = res_block(res_3, n_chns=self.nc0_g[3])

        conv_5 = Conv(n_output_chns=self.nc0_g[4],
                      kernel_size=self.k_conv,
                      acti_func=self.acti_func)(res_4, is_training)
        # TODO: fc intialisation, and compatible 2d version?
        affine = FC(n_output_chns=12)(conv_5)
        spatial_shape = moving_image.get_shape().as_list()[1:-1]
        grid_global = Grid(
            source_shape=spatial_shape, output_shape=spatial_shape)(affine)
        moving_label_global = Resampler(
            interpolation='linear',
            boundary='replicate')(moving_image, grid_global)
        return moving_label_global

