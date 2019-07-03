# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.engine.application_initializer import GlorotUniform
from niftynet.layer.convolution import ConvolutionalLayer as Conv
from niftynet.layer.downsample_res_block import DownBlock as DownRes
from niftynet.layer.fully_connected import FullyConnectedLayer as FC
from niftynet.layer.grid_warper import AffineGridWarperLayer as Grid
from niftynet.layer.layer_util import infer_spatial_rank
from niftynet.layer.linear_resize import LinearResizeLayer as Resize
from niftynet.network.base_net import BaseNet


class INetAffine(BaseNet):
    """
    ### Description
    This network estimates affine transformations from
        a pair of moving and fixed image:

            Hu et al., Label-driven weakly-supervised learning for
            multimodal deformable image registration, arXiv:1711.01666
            https://arxiv.org/abs/1711.01666

            Hu et al., Weakly-Supervised Convolutional Neural Networks for
            Multimodal Image Registration, Medical Image Analysis (2018)
            https://doi.org/10.1016/j.media.2018.07.002

        see also:
            https://github.com/YipengHu/label-reg

    ### Building blocks
    [DOWN CONV]         - Convolutional layer + Residual Unit + Max pooling
    [CONV]              - Convolutional layer
    [FC]                - Fully connected layer, outputs the affine matrix
    [WARPER]            - Grid resampling with the obtained affine matrix

    ### Diagram

    INPUT PAIR --> [DOWN CONV]x4 --> [CONV] --> [FC] --> [WARPER] --> DISPLACEMENT FIELD

    ### Constraints
    - input spatial rank should be either 2 or 3 (2D or 3D images only)

    """
    def __init__(self,
                 decay=1e-6,
                 affine_w_initializer=None,
                 affine_b_initializer=None,
                 acti_func='relu',
                 name='inet-affine'):
        """

        :param decay: float, regularisation decay
        :param affine_w_initializer: weight initialisation for affine registration network
        :param affine_b_initializer: bias initialisation for affine registration network
        :param acti_func: activation function to use
        :param name: layer name
        """

        BaseNet.__init__(self, name=name)

        self.fea = [4, 8, 16, 32, 64]
        self.k_conv = 3
        self.affine_w_initializer = affine_w_initializer
        self.affine_b_initializer = affine_b_initializer
        self.res_param = {
            'w_initializer': GlorotUniform.get_instance(''),
            'w_regularizer': regularizers.l2_regularizer(decay),
            'acti_func': acti_func}
        self.affine_param = {
            'w_regularizer': regularizers.l2_regularizer(decay),
            'b_regularizer': None}

    def layer_op(self,
                 fixed_image,
                 moving_image,
                 is_training=True,
                 **unused_kwargs):
        """

        :param fixed_image: tensor, fixed image for registration (defines reference space)
        :param moving_image: tensor, moving image to be registered to fixed
        :param is_training: boolean, True if network is in training mode
        :return: displacement fields transformed by estimating affine
        """

        spatial_rank = infer_spatial_rank(moving_image)
        spatial_shape = fixed_image.get_shape().as_list()[1:-1]

        # resize the moving image to match the fixed
        moving_image = Resize(spatial_shape)(moving_image)
        img = tf.concat([moving_image, fixed_image], axis=-1)
        res_1 = DownRes(self.fea[0], kernel_size=7, **self.res_param)(img, is_training)[0]
        res_2 = DownRes(self.fea[1], **self.res_param)(res_1, is_training)[0]
        res_3 = DownRes(self.fea[2], **self.res_param)(res_2, is_training)[0]
        res_4 = DownRes(self.fea[3], **self.res_param)(res_3, is_training)[0]

        conv_5 = Conv(n_output_chns=self.fea[4],
                      kernel_size=self.k_conv,
                      with_bias=False, feature_normalization='batch',
                      **self.res_param)(res_4, is_training)

        if spatial_rank == 2:
            affine_size = 6
        elif spatial_rank == 3:
            affine_size = 12
        else:
            tf.logging.fatal('Not supported spatial rank')
            raise NotImplementedError

        if self.affine_w_initializer is None:
            self.affine_w_initializer = init_affine_w()
        if self.affine_b_initializer is None:
            self.affine_b_initializer = init_affine_b(spatial_rank)
        affine = FC(n_output_chns=affine_size, feature_normalization=None,
                    w_initializer=self.affine_w_initializer,
                    b_initializer=self.affine_b_initializer,
                    **self.affine_param)(conv_5)
        grid_global = Grid(source_shape=spatial_shape,
                           output_shape=spatial_shape)(affine)
        return grid_global


def init_affine_w(std=1e-8):
    """

    :param std: float, standard deviation of normal distribution for weight initialisation
    :return: random weight initialisation from normal distribution with zero mean
    """
    return tf.random_normal_initializer(0, std)


def init_affine_b(spatial_rank, initial_bias=0.0):
    """

    :param spatial_rank: int, rank of inputs (either 2D or 3D)
    :param initial_bias: float, initial bias
    :return: bias initialisation for the affine matrix
    """
    if spatial_rank == 2:
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]]).flatten()
    elif spatial_rank == 3:
        identity = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0]]).flatten()
    else:
        tf.logging.fatal('Not supported spatial rank')
        raise NotImplementedError
    identity = identity.reshape([1, -1])
    identity = np.tile(identity, [1, 1])
    return tf.constant_initializer(identity + initial_bias)
