# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.engine.application_initializer import GlorotUniform
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.downsample_res_block import DownBlock as DownRes
from niftynet.layer.grid_warper import _create_affine_features
from niftynet.layer.layer_util import infer_spatial_rank, check_spatial_dims
from niftynet.layer.linear_resize import LinearResizeLayer as Resize
from niftynet.layer.spatial_gradient import SpatialGradientLayer as ImgGrad
from niftynet.layer.upsample_res_block import UpBlock as UpRes
from niftynet.network.base_net import BaseNet


class INetDense(BaseNet):
    """
     ### Description
        The network estimates dense displacement fields from a pair
        of moving and fixed images:

            Hu et al., Label-driven weakly-supervised learning for
            multimodal deformable image registration, arXiv:1711.01666
            https://arxiv.org/abs/1711.01666

            Hu et al., Weakly-Supervised Convolutional Neural Networks for
            Multimodal Image Registration, Medical Image Analysis (2018)
            https://doi.org/10.1016/j.media.2018.07.002

        see also:
            https://github.com/YipengHu/label-reg

     ### Building blocks
     [DOWN CONV]         - Convolutional layer + Residual Unit + Downsampling (Max pooling)
     [CONV]              - Convolutional layer
     [UP CONV]           - Upsampling + Sum + Residual Unit
     [FUSION]            - Multi-scale displacement fields fusion
     [DISPtoDEF]         - (Smoothing if required) Conversion to deformation field (adding base grid)


     ### Diagram
     INPUT PAIR -->  [DOWN CONV]              [UP CONV] --> [CONV] --[FUSION] --> [DISPtoDEF] --> DENSE FIELD
                         |                        |                     |
                     [DOWN CONV]              [UP CONV] --> [CONV] -----|
                         |                        |                     |
                     [DOWN CONV]              [UP CONV] --> [CONV] -----|
                         |                        |                     |
                     [DOWN CONV]              [UP CONV] --> [CONV] -----|
                          |                       |                     |
                          -------- [CONV]------------------ [CONV]-------


     ### Constraints
        - input spatial rank should be either 2 or 3 (2D or 3D images only)
        - fixed image size should be divisible by 16
    """
    def __init__(self,
                 decay=0.0,
                 smoothing=0,
                 disp_w_initializer=None,
                 disp_b_initializer=None,
                 acti_func='relu',
                 multi_scale_fusion=True,
                 name='inet-dense'):
        """

        :param decay: float, regularisation decay
        :param smoothing: float, smoothing factor for dense displacement field
        :param disp_w_initializer: initialisation of the displacement fields
        :param disp_b_initializer: initialisation of the displacement fields
        :param acti_func: activation function to use
        :param multi_scale_fusion: True/False indicating whether to use
            multiscale feature fusion.
        :param name: layer name
        """
        BaseNet.__init__(self, name=name)

        # self.fea = [40, 80, 160, 320, 640]
        # self.fea = [32, 64, 128, 256, 512]
        self.fea = [30, 60, 120, 240, 480]
        # self.fea = [16, 32, 64, 128, 256]
        self.k_conv = 3
        self.multi_scale_fusion = multi_scale_fusion

        self.down_res_param = {
            'w_initializer': GlorotUniform.get_instance(''),
            'w_regularizer': regularizers.l2_regularizer(decay),
            'acti_func': acti_func}

        self.up_res_param = {
            'acti_func': acti_func,
            'w_initializer': GlorotUniform.get_instance(''),
            'w_regularizer': regularizers.l2_regularizer(decay),
            'is_residual_upsampling': True,
            'type_string': 'bn_acti_conv'}

        # displacement initialiser & regulariser
        if disp_w_initializer is None:
            disp_b_initializer = tf.constant_initializer(0.0)
            #disp_w_initializer = tf.random_normal_initializer(0, 1e-4)
        if disp_b_initializer is None:
            disp_b_initializer = tf.constant_initializer(0.0)
            #disp_w_initializer = tf.random_normal_initializer(0, 0.0)
        self.disp_param = {
            'w_initializer': disp_w_initializer,
            'w_regularizer': regularizers.l2_regularizer(decay),
            'b_initializer': disp_b_initializer,
            'b_regularizer': None}

        if smoothing > 0:
            self.smoothing_func = _smoothing_func(smoothing)
        else:
            self.smoothing_func = None

    def layer_op(self,
                 fixed_image,
                 moving_image,
                 base_grid=None,
                 is_training=True,
                 **unused_kwargs):
        """

        :param fixed_image: tensor, fixed image for registration (defines reference space)
        :param moving_image: tensor, moving image to be registered to fixed
        :param base_grid: initial identity or affine displacement field
        :param is_training: boolean, True if network is in training mode
        :return: estimated dense displacement fields
        """

        spatial_rank = infer_spatial_rank(fixed_image)
        spatial_shape = fixed_image.get_shape().as_list()[1:-1]
        check_spatial_dims(fixed_image, lambda x: x % 16 == 0)

        #  resize the moving image to match the fixed
        moving_image = Resize(spatial_shape)(moving_image)
        img = tf.concat([moving_image, fixed_image], axis=-1)
        down_res_0, conv_0_0, _ = \
            DownRes(self.fea[0], kernel_size=7, **self.down_res_param)(img, is_training)
        down_res_1, conv_0_1, _ = \
            DownRes(self.fea[1], **self.down_res_param)(down_res_0, is_training)
        down_res_2, conv_0_2, _ = \
            DownRes(self.fea[2], **self.down_res_param)(down_res_1, is_training)
        down_res_3, conv_0_3, _ = \
            DownRes(self.fea[3], **self.down_res_param)(down_res_2, is_training)

        conv_4 = Conv(n_output_chns=self.fea[4],
                      kernel_size=self.k_conv,
                      **self.down_res_param)(down_res_3, is_training)

        up_res_0 = UpRes(self.fea[3], **self.up_res_param)(
            conv_4, conv_0_3, is_training)
        up_res_1 = UpRes(self.fea[2], **self.up_res_param)(
            up_res_0, conv_0_2, is_training)
        up_res_2 = UpRes(self.fea[1], **self.up_res_param)(
            up_res_1, conv_0_1, is_training)
        up_res_3 = UpRes(self.fea[0], **self.up_res_param)(
            up_res_2, conv_0_0, is_training)

        if self.multi_scale_fusion:
            output_list = [up_res_3, up_res_2, up_res_1, up_res_0, conv_4]
        else:
            output_list = [up_res_3]

        # converting all output layers to displacement fields 
        dense_fields = []
        for scale_out in output_list:
            field = Conv(n_output_chns=spatial_rank,
                         kernel_size=self.k_conv,
                         with_bias=True,
                         feature_normalization=None,
                         acti_func=None,
                         **self.disp_param)(scale_out)
            resized_field = Resize(new_size=spatial_shape)(field)
            dense_fields.append(resized_field)

        if base_grid is None:
            # adding a reference grid if it doesn't exist
            in_spatial_size = [None] * spatial_rank
            base_grid = _create_affine_features(output_shape=spatial_shape,
                                                source_shape=in_spatial_size)
            base_grid = np.asarray(base_grid[:-1])
            base_grid = np.reshape(
                base_grid.T, [-1] + spatial_shape + [spatial_rank])
            base_grid = tf.constant(base_grid, dtype=resized_field.dtype)

        if self.multi_scale_fusion and len(dense_fields) > 1:
            dense_field = tf.reduce_sum(dense_fields, axis=0)
        else:
            dense_field = dense_fields[0]

        # TODO filtering
        if self.smoothing_func is not None:
            dense_field = self.smoothing_func(dense_field, spatial_rank)

        tf.add_to_collection('bending_energy',
                             _computing_bending_energy(dense_field))
        tf.add_to_collection('gradient_norm',
                             _computing_gradient_norm(dense_field))

        dense_field = dense_field + base_grid
        return dense_field


def _get_smoothing_kernel(sigma, spatial_rank):
    """

    :param sigma: float, standard deviation for gaussian smoothing kernel
    :param spatial_rank: int, rank of input
    :return: smoothing kernel
    """
    # sigma defined in voxel not in freeform deformation grid
    if sigma <= 0:
        raise NotImplementedError
    tail = int(sigma * 2)
    if spatial_rank == 2:
        x, y = np.mgrid[-tail:tail + 1, -tail:tail + 1]
        g = np.exp(-0.5 * (x * x + y * y) / sigma * sigma)
    elif spatial_rank == 3:
        x, y, z = np.mgrid[-tail:tail + 1, -tail:tail + 1, -tail:tail + 1]
        g = np.exp(-0.5 * (x * x + y * y + z * z) / sigma * sigma)
    else:
        raise NotImplementedError
    return g / g.sum()


def _smoothing_func(sigma):
    def smoothing(dense_field, spatial_rank):
        """

        :param dense_field: tensor, dense field to be smoothed
        :param spatial_rank: int, rank of input images
        :return: smoothed dense field
        """
        kernel = _get_smoothing_kernel(sigma, spatial_rank)
        kernel = tf.constant(kernel, dtype=dense_field.dtype)
        kernel = tf.expand_dims(kernel, axis=-1)
        kernel = tf.expand_dims(kernel, axis=-1)
        smoothed = [
            tf.nn.convolution(tf.expand_dims(coord, axis=-1), kernel, 'SAME')
            for coord in tf.unstack(dense_field, axis=-1)]
        return tf.concat(smoothed, axis=-1)

    return smoothing


def _computing_bending_energy(displacement):
    """

    :param displacement: tensor, displacement field
    :return: bending energy
    """
    spatial_rank = infer_spatial_rank(displacement)
    if spatial_rank == 2:
        return _computing_bending_energy_2d(displacement)
    if spatial_rank == 3:
        return _computing_bending_energy_3d(displacement)
    raise NotImplementedError(
        "Not implmented: bending energy for {}-d input".format(spatial_rank))


def _computing_bending_energy_2d(displacement):
    """

    :param displacement: 2D tensor, displacement field
    :return: bending energy
    """
    dTdx = ImgGrad(spatial_axis=0)(displacement)
    dTdy = ImgGrad(spatial_axis=1)(displacement)

    dTdxx = ImgGrad(spatial_axis=0)(dTdx)
    dTdyy = ImgGrad(spatial_axis=1)(dTdy)
    dTdxy = ImgGrad(spatial_axis=1)(dTdx)

    energy = tf.reduce_mean([dTdxx * dTdxx, dTdyy * dTdyy, 2 * dTdxy * dTdxy])
    return energy


def _computing_bending_energy_3d(displacement):
    """

    :param displacement: 3D tensor, displacement field
    :return: bending energy
    """
    dTdx = ImgGrad(spatial_axis=0)(displacement)
    dTdy = ImgGrad(spatial_axis=1)(displacement)
    dTdz = ImgGrad(spatial_axis=2)(displacement)

    dTdxx = ImgGrad(spatial_axis=0)(dTdx)
    dTdyy = ImgGrad(spatial_axis=1)(dTdy)
    dTdzz = ImgGrad(spatial_axis=2)(dTdz)

    dTdxy = ImgGrad(spatial_axis=1)(dTdx)
    dTdyz = ImgGrad(spatial_axis=2)(dTdy)
    dTdxz = ImgGrad(spatial_axis=2)(dTdx)

    energy = tf.reduce_mean(
        [dTdxx * dTdxx, dTdyy * dTdyy, dTdzz * dTdzz,
         2 * dTdxy * dTdxy, 2 * dTdxz * dTdxz, 2 * dTdyz * dTdyz])
    return energy


def _computing_gradient_norm(displacement, flag_L1=False):
    """

    :param displacement: tensor, displacement field
    :param flag_L1: boolean, True if L1 norm shoudl be used
    :return: L2 (or L1) norm of gradients
    """
    norms = []
    for spatial_ind in range(infer_spatial_rank(displacement)):
        dTdt = ImgGrad(spatial_axis=spatial_ind)(displacement)
        if flag_L1:
            norms.append(tf.abs(dTdt))
        else:
            norms.append(dTdt * dTdt)
    return tf.reduce_mean(norms)
