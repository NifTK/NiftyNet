# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.layer.layer_util import infer_spatial_rank


class SpatialGradientLayer(Layer):
    """
    Computing image spatial gradients.
    """

    def __init__(self,
                 spatial_axis=0,
                 do_cropping=True,
                 name='spatial_gradient'):

        Layer.__init__(self, name=name)
        self.spatial_axis = int(spatial_axis)
        self.do_cropping = do_cropping

    def layer_op(self, input_tensor):
        """
        Computing spatial gradient of ``input_tensor`` along
        ``self.spatial_axis``.

        output is equivalent to convolve along ``spatial_axis`` with a
         kernel: ``[-1, 0, 1]``

        This layer assumes the first and the last dimension of the input
        tensor represent batch and feature channels.
        Therefore ``spatial_axis=1`` is computing gradient along the
        third dimension of input tensor, i.e., ``input_tensor[:, :, y, ...]``

        Given the input with shape ``[B, X, Y, Z, C]``, and ``spatial_axis=1``
        the output shape is::
            [B, X-2, Y-2, Z-2, C] if do_scropping is True
            [B, X, Y-2, Z, C] otherwise

        Setting do_cropping to True makes the output tensor has the same
        dimensionality for different ``spatial_axis``.

        :param input_tensor: a batch of images with a shape of
            ``[Batch, x[, y, z, ... ], Channel]``
        :return: spatial gradients of ``input_tensor``
        """

        spatial_rank = infer_spatial_rank(input_tensor)
        spatial_size = input_tensor.get_shape().as_list()[1:-1]
        if self.do_cropping:
            # remove two elements in all spatial dims
            spatial_size = [size_x - 2 for size_x in spatial_size]
            spatial_begins = [1] * spatial_rank
        else:
            # remove two elements along the gradient dim only
            spatial_size[self.spatial_axis] = spatial_size[self.spatial_axis] -2
            spatial_begins = [0] * spatial_rank

        spatial_begins[self.spatial_axis] = 2
        begins_0 = [0] + spatial_begins + [0]

        spatial_begins[self.spatial_axis] = 0
        begins_1 = [0] + spatial_begins + [0]

        sizes_0 = [-1] + spatial_size + [-1]
        sizes_1 = [-1] + spatial_size + [-1]

        image_gradients = \
            tf.slice(input_tensor, begins_0, sizes_0) - \
            tf.slice(input_tensor, begins_1, sizes_1)
        return image_gradients
