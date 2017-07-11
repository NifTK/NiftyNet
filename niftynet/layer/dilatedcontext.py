# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer import layer_util


class DilatedTensor(object):
    """
    This context manager makes a wrapper of input_tensor
    When created, the input_tensor is dilated,
    the input_tensor resumes to original space when exiting the context.
    """

    def __init__(self, input_tensor, dilation_factor):
        assert (layer_util.check_spatial_dims(
            input_tensor, lambda x: x % dilation_factor == 0))
        self._tensor = input_tensor
        self.dilation_factor = dilation_factor
        # parameters to transform input tensor
        self.spatial_rank = layer_util.infer_spatial_rank(self._tensor)
        self.zero_paddings = [[0, 0]] * self.spatial_rank
        self.block_shape = [dilation_factor] * self.spatial_rank

    def __enter__(self):
        if self.dilation_factor > 1:
            self._tensor = tf.space_to_batch_nd(self._tensor,
                                                self.block_shape,
                                                self.zero_paddings,
                                                name='dilated')
        return self

    def __exit__(self, *args):
        if self.dilation_factor > 1:
            self._tensor = tf.batch_to_space_nd(self._tensor,
                                                self.block_shape,
                                                self.zero_paddings,
                                                name='de-dilate')

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, value):
        self._tensor = value
