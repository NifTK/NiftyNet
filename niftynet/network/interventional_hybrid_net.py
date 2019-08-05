# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.resampler import ResamplerLayer as resampler
from niftynet.network.base_net import BaseNet
from niftynet.network.interventional_affine_net import INetAffine
from niftynet.network.interventional_dense_net import INetDense


class INetHybridPreWarp(BaseNet):
    """
    ### Description
    Re-implementation of the registration network proposed in:

            Hu et al., Label-driven weakly-supervised learning for
            multimodal deformable image registration, arXiv:1711.01666
            https://arxiv.org/abs/1711.01666

            Hu et al., Weakly-Supervised Convolutional Neural Networks for
            Multimodal Image Registration, Medical Image Analysis (2018)
            https://doi.org/10.1016/j.media.2018.07.002

        see also:
            https://github.com/YipengHu/label-reg

    ### Building blocks
    [GLOBAL]            - INetAffine from interventional_affine_net.py
    [RESAMPLER]         - Layer to resample the moving image with estimated affine
    [DENSE]             - INetDense from intervetional_dense_net.py

    ### Diagram

    INPUT PAIR --> [GLOBAL]  --> [RESAMPLER] --> [DENSE] --> DENSE FIELD, AFFINE FIELD

    ### Constraints
        - input spatial rank should be either 2 or 3 (2D or 3D images only)
        - fixed image size should be divisible by 16
    """
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

        :param decay: float, regularisation decay
        :param affine_w_initializer: weight initialisation for affine registration network
        :param affine_b_initializer: bias initialisation for affine registration network
        :param disp_w_initializer: weight initialisation for dense registration network
        :param disp_b_initializer: bias initialisation for dense registration network
        :param acti_func: activation function to use
        :param interp: string, type of interpolation for the resampling [default:linear]
        :param boundary: string, padding mode to deal with image boundary
        :param name: layer name
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

    def layer_op(self,
                 fixed_image,
                 moving_image,
                 is_training=True,
                 **unused_kwargs):
        """

        :param fixed_image: tensor, fixed image for registration (defines reference space)
        :param moving_image: tensor, moving image to be registered to fixed
        :param is_training: boolean, True if network is in training mode
        :param unused_kwargs: not in use
        :return: estimated final dense and affine displacement fields
        """
        affine_field = self.global_net(fixed_image, moving_image, is_training)
        moving_image = resampler(
            interpolation=self.interp,
            boundary=self.boundary)(moving_image, affine_field)
        dense_field = self.local_net(
            fixed_image, moving_image, affine_field, is_training)
        return dense_field, affine_field


class INetHybridTwoStream(BaseNet):
    """
    ### Description
    Re-implementation of the registration network proposed in:

            Hu et al., Label-driven weakly-supervised learning for
            multimodal deformable image registration, arXiv:1711.01666
            https://arxiv.org/abs/1711.01666

            Hu et al., Weakly-Supervised Convolutional Neural Networks for
            Multimodal Image Registration, Medical Image Analysis (2018)
            https://doi.org/10.1016/j.media.2018.07.002

        see also:
            https://github.com/YipengHu/label-reg

    ### Building blocks
    [GLOBAL]            - INetAffine from interventional_affine_net.py
    [DENSE]             - INetDense from intervetional_dense_net.py

    ### Diagram

    INPUT PAIR --> [GLOBAL] --> AFFINE FIELD --- DENSE + AFFINE FIELD
         |                                       |
          -------> [DENSE] --> DENSE FIELD ------

    ### Constraints
        - input spatial rank should be either 2 or 3 (2D or 3D images only)
        - fixed image size should be divisible by 16
    """
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
        """

        :param decay: float, regularisation decay
        :param affine_w_initializer: weight initialisation for affine registration network
        :param affine_b_initializer: bias initialisation for affine registration network
        :param disp_w_initializer: weight initialisation for dense registration network
        :param disp_b_initializer: bias initialisation for dense registration network
        :param acti_func: activation function to use
        :param interp: string, type of interpolation for the resampling [default:linear] - not in use
        :param boundary: string, padding mode to deal with image boundary [default: replicate] - not is use
        :param name: layer name
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

    def layer_op(self,
                 fixed_image,
                 moving_image,
                 is_training=True,
                 **unused_kwargs):
        """

        :param fixed_image: tensor, fixed image for registration (defines reference space)
        :param moving_image: tensor, moving image to be registered to fixed
        :param is_training: boolean, True if network is in training mode
        :param unused_kwargs: not in use
        :return: estimated total, dense and affine displacement fields
        """
        affine_field = self.global_net(fixed_image, moving_image, is_training)
        dense_field = self.local_net(fixed_image, moving_image, is_training)
        return dense_field + affine_field, dense_field, affine_field
