# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvLayer
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_OP = set(['SUM', 'CONCAT'])


class ElementwiseLayer(TrainableLayer):
    """
    This class takes care of the elementwise sum in a residual connection
    It matches the channel dims from two branch flows,
    by either padding or projection if necessary.
    """

    def __init__(self,
                 func,
                 initializer=None,
                 regularizer=None,
                 name='residual'):

        self.func = look_up_operations(func.upper(), SUPPORTED_OP)
        self.layer_name = '{}_{}'.format(name, self.func.lower())

        super(ElementwiseLayer, self).__init__(name=self.layer_name)
        self.initializers = {'w': initializer}
        self.regularizers = {'w': regularizer}

    def layer_op(self, param_flow, bypass_flow):
        n_param_flow = param_flow.shape[-1]
        n_bypass_flow = bypass_flow.shape[-1]
        spatial_rank = layer_util.infer_spatial_rank(param_flow)

        output_tensor = param_flow
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
                                      w_initializer=self.initializers['w'],
                                      w_regularizer=self.regularizers['w'],
                                      name='proj')
                bypass_flow = projector(bypass_flow)

            # element-wise sum of both paths
            output_tensor = param_flow + bypass_flow

        elif self.func == 'CONCAT':
            output_tensor = tf.concat([param_flow, bypass_flow], axis=-1)

        return output_tensor
