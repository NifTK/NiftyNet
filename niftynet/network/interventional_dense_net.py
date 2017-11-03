# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.engine.application_initializer import GlorotUniform
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.downsample_res_block import DownBlock as DownRes
from niftynet.layer.layer_util import infer_spatial_rank
from niftynet.layer.upsample_res_block import UpBlock as UpRes
from niftynet.network.base_net import BaseNet


class INetDense(BaseNet):
    def __init__(self,
                 decay,
                 disp_w_initializer=None,
                 disp_b_initializer=None,
                 acti_func='relu',
                 name='inet-dense'):
        BaseNet.__init__(self, name=name)

        # self.fea = [32, 64, 128, 256, 512]
        self.fea = [4, 8, 16, 32, 64]
        self.k_conv = 3
        self.res_param = {
            'w_initializer': GlorotUniform.get_instance(''),
            'w_regularizer': regularizers.l2_regularizer(decay),
            'acti_func': acti_func}
        self.disp_param = {
            'w_initializer': disp_w_initializer,
            'w_regularizer': regularizers.l2_regularizer(decay),
            'b_initializer': disp_b_initializer,
            'b_regularizer': None}

    def layer_op(self, fixed_image, moving_image, is_training=True):
        """
        returns estimated dense displacement fields
        """
        img = tf.concat([moving_image, fixed_image], axis=-1)
        down_res_0, conv_0_0 = \
            DownRes(self.fea[0], **self.res_param)(img, is_training)
        down_res_1, conv_0_1 = \
            DownRes(self.fea[1], **self.res_param)(down_res_0, is_training)
        down_res_2, conv_0_2 = \
            DownRes(self.fea[2], **self.res_param)(down_res_1, is_training)
        down_res_3, conv_0_3 = \
            DownRes(self.fea[3], **self.res_param)(down_res_2, is_training)

        conv_4 = Conv(n_output_chns=self.fea[4],
                      kernel_size=self.k_conv,
                      **self.res_param)(down_res_3, is_training)

        up_res_0 = UpRes(self.fea[3], **self.res_param)(
            conv_4, conv_0_3, is_training)
        up_res_1 = UpRes(self.fea[2], **self.res_param)(
            up_res_0, conv_0_2, is_training)
        up_res_2 = UpRes(self.fea[1], **self.res_param)(
            up_res_1, conv_0_1, is_training)
        up_res_3 = UpRes(self.fea[0], **self.res_param)(
            up_res_2, conv_0_0, is_training)

        spatial_rank = infer_spatial_rank(moving_image)
        conv_5 = Conv(n_output_chns=spatial_rank,
                      kernel_size=self.k_conv,
                      with_bias=True,
                      with_bn=False,
                      acti_func=None,
                      **self.disp_param)(up_res_3)
        # TODO filtering
        return conv_5
