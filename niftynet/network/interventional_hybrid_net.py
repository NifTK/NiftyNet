# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.downsample_res_block import DownBlock as DownRes
from niftynet.layer.fully_connected import FullyConnectedLayer as FC
from niftynet.layer.grid_warper import AffineGridWarperLayer as Grid
from niftynet.layer.layer_util import infer_spatial_rank
from niftynet.layer.upsample_res_block import UpBlock as UpRes
from niftynet.network.base_net import BaseNet
from niftynet.network.interventional_affine_net import INetAffine
from niftynet.network.interventional_dense_net import INetDense
from niftynet.layer.resampler import ResamplerLayer as resampler

class INetHybrid(BaseNet):
    def __init__(self,
                 decay,
                 affine_w_initializer=None,
                 affine_b_initializer=None,
                 disp_w_initializer=None,
                 disp_b_initializer=None,
                 acti_func='relu',
                 interp='linear',
                 boundary='replicate',
                 name='inet-hybrid'):
        BaseNet.__init__(self, name=name)
        self.global_net = INetAffine(decay=decay,
                                     affine_w_initializer=affine_w_initializer,
                                     affine_b_initializer=affine_b_initializer,
                                     acti_func=acti_func,
                                     interp=interp,
                                     boundary=boundary,
                                     name='inet-global')
        self.local_net = INetDense(decay=decay,
                                   disp_w_initializer=disp_w_initializer,
                                   disp_b_initializer=disp_b_initializer,
                                   acti_func=acti_func,
                                   name='inet-local')
        self.interp = interp
        self.boundary = boundary

    def layer_op(self, fixed_image, moving_image, is_training=True):
        affine_field = self.global_net(fixed_image, moving_image, is_training)
        moving_image = resampler(
            interpolation=self.interp,
            boundary=self.boundary)(moving_image, affine_field)
        dense_field = self.local_net(fixed_image, moving_image, is_training)
        return affine_field, dense_field
