import numpy as np

import tensorflow as tf
from base import Layer
from convolution import ConvLayer
import layer_util


class ElementwiseSumLayer(Layer):
    """
    This class takes care of the elementwise sum in a residual connection
    It matches the channel dims from two branch flows,
    by either padding or projection if necessary.
    """
    def __init__(self, initializer=None, regularizer=None, name='residual'):
        super(ElementwiseSumLayer, self).__init__(name=name)
        self.initializer = initializer
        self.regularizer = regularizer

    def layer_op(self, param_flow, bypass_flow):
        n_param_flow = param_flow.get_shape()[-1]
        n_bypass_flow = bypass_flow.get_shape()[-1]
        spatial_rank = layer_util.infer_spatial_rank(param_flow)

        if n_param_flow == n_bypass_flow:
            output_tensor = param_flow + bypass_flow

        elif n_param_flow > n_bypass_flow:  # pad the channel dim
            pad_1 = np.int((n_param_flow - n_bypass_flow) // 2)
            pad_2 = np.int(n_param_flow - n_bypass_flow - pad_1)
            padding_dims = np.vstack(([[0, 0]],
                                      [[0, 0]] * spatial_rank,
                                      [[pad_1, pad_2]]))
            padded_bypass_flow = tf.pad(tensor=bypass_flow,
                                      paddings=padding_dims.tolist(),
                                      mode='CONSTANT')
            output_tensor = param_flow + padded_bypass_flow

        elif n_param_flow < n_bypass_flow:  # make a projection to the lower dim
            projector = ConvLayer(n_output_chns=n_bypass_flow,
                                  kernel_size=1,
                                  stride=1,
                                  padding='SAME',
                                  w_initializer=self.initializer,
                                  w_regularizer=self.regularizer,
                                  name='proj')
            proj_param_flow = projector(param_flow)
            output_tensor = proj_param_flow + bypass_flow

        return output_tensor
