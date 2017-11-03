# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.activation import ActiLayer as Acti
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.deconvolution import DeconvolutionalLayer as Deconv
from niftynet.layer.downsample import DownSampleLayer as Down
from niftynet.layer.fully_connected import FullyConnectedLayer as FC
from niftynet.layer.grid_warper import AffineGridWarperLayer as Grid
from niftynet.layer.resampler import ResamplerLayer as Resampler
from niftynet.network.base_net import BaseNet
from niftynet.layer.layer_util import infer_spatial_rank


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
                         b_initializer=b_initializer,
                         b_regularizer=b_regularizer,
                         acti_func=acti_func,
                         name=name)
        # TODO initializer
        self.n_features = [4, 8, 16, 32, 64]
        self.k_conv = 3
        self.interp = 'linear'
        self.boundary = 'replicate'

    def layer_op(self,
                 fixed_image, fixed_label, moving_image, moving_label,
                 displacement=None, is_training=True):
        images = tf.concat([moving_image, fixed_image], axis=-1)

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

        res_1 = res_block(images, n_chns=self.n_features[0])
        res_2 = res_block(res_1, n_chns=self.n_features[1])
        res_3 = res_block(res_2, n_chns=self.n_features[2])
        res_4 = res_block(res_3, n_chns=self.n_features[3])

        conv_5 = Conv(n_output_chns=self.n_features[4],
                      kernel_size=self.k_conv,
                      acti_func=self.acti_func)(res_4, is_training)

        # TODO: fc initialisation, and compatible 2d version?
        spatial_shape = moving_image.get_shape().as_list()[1:-1]
        if len(spatial_shape) == 2:
            affine = FC(n_output_chns=6, with_bn=False)(conv_5)
        elif len(spatial_shape) == 3:
            affine = FC(n_output_chns=12, with_bn=False)(conv_5)
        else:
            tf.logging.fatal('Not supported spatial rank')
            raise NotImplementedError
        grid_global = Grid(source_shape=spatial_shape,
                           output_shape=spatial_shape)(affine)
        #moving_image_global = Resampler(
        #    interpolation=self.interp,
        #    boundary=self.boundary)(moving_image, grid_global)
        return grid_global


class INetLocal(BaseNet):
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='inet-local'):
        BaseNet.__init__(self,
                         num_classes,
                         w_initializer=w_initializer,
                         w_regularizer=w_regularizer,
                         b_initializer=b_initializer,
                         b_regularizer=b_regularizer,
                         acti_func=acti_func,
                         name=name)

        #self.n_features = [32, 64, 128, 256, 512]
        self.n_features = [4, 8, 16, 32, 64]
        self.k_conv = 3

    def layer_op(self,
                 fixed_image, fixed_label, moving_image, moving_label,
                 displacement=None, is_training=True):
        spatial_rank = infer_spatial_rank(moving_image)
        images = tf.concat([moving_image, fixed_image], axis=-1)

        def res_block(inputs, n_chns=4, down_sampling=True, forwarding=None):
            if down_sampling:
                conv_0 = Conv(n_output_chns=n_chns,
                              kernel_size=self.k_conv,
                              acti_func=self.acti_func)(inputs, is_training)
            else:  # do upsampling
                conv_0 = Deconv(n_output_chns=n_chns,
                                kernel_size=self.k_conv,
                                stride=2,
                                acti_func=self.acti_func)(inputs, is_training)
                if forwarding is not None:
                    conv_0 = conv_0 + forwarding
            conv_1 = Conv(n_output_chns=n_chns,
                          kernel_size=self.k_conv,
                          acti_func=self.acti_func)(conv_0, is_training)
            conv_2 = Conv(n_output_chns=n_chns,
                          kernel_size=self.k_conv,
                          acti_func=None)(conv_1, is_training)
            conv_res = Acti(self.acti_func)(conv_2 + conv_0)
            if down_sampling:
                return Down('MAX', kernel_size=2, stride=2)(conv_res), conv_0
            return conv_res

        down_res_0, conv_0_0 = res_block(images, self.n_features[0])
        down_res_1, conv_0_1 = res_block(down_res_0, self.n_features[1], True)
        down_res_2, conv_0_2 = res_block(down_res_1, self.n_features[2], True)
        down_res_3, conv_0_3 = res_block(down_res_2, self.n_features[3], True)

        conv_4 = Conv(n_output_chns=self.n_features[4],
                      kernel_size=self.k_conv,
                      acti_func=self.acti_func)(down_res_3, is_training)

        up_res_0 = res_block(conv_4, self.n_features[3], False, conv_0_3)
        up_res_1 = res_block(up_res_0, self.n_features[2], False, conv_0_2)
        up_res_2 = res_block(up_res_1, self.n_features[1], False, conv_0_1)
        up_res_3 = res_block(up_res_2, self.n_features[0], False, conv_0_0)

        conv_5 = Conv(n_output_chns=spatial_rank,
                      kernel_size=self.k_conv,
                      with_bias=True,
                      with_bn=False)(up_res_3)
        return conv_5


class INetComposite(BaseNet):
    pass
