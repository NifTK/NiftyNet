# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.engine.application_initializer import GlorotUniform
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.downsample_res_block import DownBlock as DownRes
from niftynet.layer.fully_connected import FullyConnectedLayer as FC
from niftynet.layer.grid_warper import AffineGridWarperLayer as Grid
from niftynet.layer.layer_util import infer_spatial_rank
from niftynet.network.base_net import BaseNet


class INetAffine(BaseNet):
    def __init__(self,
                 decay=1e-6,
                 affine_w_initializer=None,
                 affine_b_initializer=None,
                 acti_func='relu',
                 interp='linear',
                 boundary='replicate',
                 name='inet-affine'):

        BaseNet.__init__(self, name=name)
        # TODO initializer, WCE changed
        self.fea = [4, 8, 16, 32, 64]
        self.k_conv = 3
        self.interp = interp
        self.boundary = boundary
        self.res_param = {'w_initializer': GlorotUniform.get_instance(''),
                          'w_regularizer': regularizers.l2_regularizer(decay),
                          'acti_func': acti_func}
        self.affine_param = {
            'w_initializer': affine_w_initializer,
            'w_regularizer': regularizers.l2_regularizer(decay),
            'b_initializer': affine_b_initializer,
            'b_regularizer': None}

    def layer_op(self, fixed_image, moving_image, is_training=True):
        """
        returns displacement fields transformed by first estimating affine
        """
        img = tf.concat([moving_image, fixed_image], axis=-1)
        res_1, _ = DownRes(self.fea[0], **self.res_param)(img, is_training)
        res_2, _ = DownRes(self.fea[1], **self.res_param)(res_1, is_training)
        res_3, _ = DownRes(self.fea[2], **self.res_param)(res_2, is_training)
        res_4, _ = DownRes(self.fea[3], **self.res_param)(res_3, is_training)

        conv_5 = Conv(n_output_chns=self.fea[4],
                      kernel_size=self.k_conv,
                      with_bias=False, with_bn=True,
                      **self.res_param)(res_4, is_training)

        # TODO: fc initialisation, and compatible 2d version?
        spatial_rank = infer_spatial_rank(moving_image)
        if spatial_rank == 2:
            affine = FC(n_output_chns=6, with_bn=False,
                        **self.affine_param)(conv_5)
        elif spatial_rank == 3:
            affine = FC(n_output_chns=12, with_bn=False,
                        **self.affine_param)(conv_5)
        else:
            tf.logging.fatal('Not supported spatial rank')
            raise NotImplementedError
        spatial_shape = moving_image.get_shape().as_list()[1:-1]
        grid_global = Grid(source_shape=spatial_shape,
                           output_shape=spatial_shape)(affine)
        return grid_global
