import numpy as np

import tensorflow as tf
from .base import Layer
from .convolution import ConvLayer
from . import layer_util

SUPPORTED_OP = set(['SUM', 'CONCAT'])


class ElementwiseLayer(Layer):
    """
    This class takes care of the elementwise sum in a residual connection
    It matches the channel dims from two branch flows,
    by either padding or projection if necessary.
    """

    def __init__(self, func, initializer=None, regularizer=None, name='residual'):
        self.func = func.upper()
        assert self.func in SUPPORTED_OP
        self.layer_name = 'res_{}'.format(self.func.lower())
        super(ElementwiseLayer, self).__init__(name=self.layer_name)
        self.initializer = initializer
        self.regularizer = regularizer

    def layer_op(self, param_flow, bypass_flow):
        n_param_flow = param_flow.get_shape()[-1]
        n_bypass_flow = bypass_flow.get_shape()[-1]
        spatial_rank = layer_util.infer_spatial_rank(param_flow)

        if self.func == 'SUM':
            if n_param_flow > n_bypass_flow:  # pad the channel dim
                pad_1 = np.int((n_param_flow - n_bypass_flow) // 2)
                pad_2 = np.int(n_param_flow - n_bypass_flow - pad_1)
                padding_dims = np.vstack(([[0, 0]],
                                          [[0, 0]] * spatial_rank,
                                          [[pad_1, pad_2]]))
                bypass_flow = tf.pad(tensor=bypass_flow,
                                     paddings=padding_dims.tolist(),
                                     mode='CONSTANT')
            elif n_param_flow < n_bypass_flow:  # make a projection
                projector = ConvLayer(n_output_chns=n_param_flow,
                                      kernel_size=1,
                                      stride=1,
                                      padding='SAME',
                                      w_initializer=self.initializer,
                                      w_regularizer=self.regularizer,
                                      name='proj')
                bypass_flow = projector(bypass_flow)

            # element-wise sum of both paths
            output_tensor = param_flow + bypass_flow

        elif self.func == 'CONCAT':
            output_tensor = tf.concat([param_flow, bypass_flow], axis=-1)

        return output_tensor
