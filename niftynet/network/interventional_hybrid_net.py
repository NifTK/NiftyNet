# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.resampler import ResamplerLayer as resampler
from niftynet.network.base_net import BaseNet
from niftynet.network.interventional_affine_net import INetAffine
from niftynet.network.interventional_dense_net import INetDense


class INetHybridPreWarp(BaseNet):
    def __init__(self,
                 decay,
                 affine_w_initializer=None,
                 affine_b_initializer=None,
                 disp_w_initializer=None,
                 disp_b_initializer=None,
                 acti_func='relu',
                 interp='linear',
                 boundary='replicate',
                 name='inet-hybrid-pre-warp'):
        """
        Re-implementation of the registration network proposed in:

            Hu et al., Label-driven weakly-supervised learning for
            multimodal deformable image registration, arXiv:1711.01666
            https://arxiv.org/abs/1711.01666

        :param decay:
        :param affine_w_initializer:
        :param affine_b_initializer:
        :param disp_w_initializer:
        :param disp_b_initializer:
        :param acti_func:
        :param interp:
        :param boundary:
        :param name:
        """
        BaseNet.__init__(self, name=name)
        self.global_net = INetAffine(decay=decay,
                                     affine_w_initializer=affine_w_initializer,
                                     affine_b_initializer=affine_b_initializer,
                                     acti_func=acti_func,
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
        dense_field = self.local_net(
            fixed_image, moving_image, affine_field, is_training)
        return dense_field, affine_field


class INetHybridTwoStream(BaseNet):
    def __init__(self,
                 decay,
                 affine_w_initializer=None,
                 affine_b_initializer=None,
                 disp_w_initializer=None,
                 disp_b_initializer=None,
                 acti_func='relu',
                 interp='linear',
                 boundary='replicate',
                 name='inet-hybrid-two-stream'):
        BaseNet.__init__(self, name=name)
        self.global_net = INetAffine(decay=decay,
                                     affine_w_initializer=affine_w_initializer,
                                     affine_b_initializer=affine_b_initializer,
                                     acti_func=acti_func,
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
        dense_field = self.local_net(fixed_image, moving_image, is_training)
        return dense_field + affine_field, dense_field, affine_field
