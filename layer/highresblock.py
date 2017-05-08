import tensorflow as tf
import numpy as np

from base import Layer
from convolution import ConvLayer
from bn import BNLayer
from activation import ActiLayer
from elementwise_sum import ElementwiseSumLayer

class HighResBlock(Layer):
    """
    This class define a high-resolution block with residual connections
    kernels - specify kernel sizes of each convolutional layer
            - e.g.: kernels=(5, 5, 5) indicate three conv layers of kernel_size 5
    with_res - whether to add residual connections to bypass the conv layers
    """
    def __init__(self,
                 n_output_chns,
                 kernels=(3, 3),
                 dilation_factor=1,
                 acti_type='relu',
                 w_initializer=None,
                 w_regularizer=None,
                 with_res=True,
                 name='HighResBlock'):
        self.n_output_chns = n_output_chns
        if hasattr(kernels, "__iter__"):  # is a list of layer kernel_sizes
            self.kernels = kernels
        else:  # is a single number (indicating single layer)
            self.kernels = [kernels]
        self.acti_type = acti_type
        self.w_initializer=None
        self.w_regularizer=None
        self.layer_name = name
        self.with_res = with_res
        super(HighResBlock, self).__init__(name=self.layer_name)

    def layer_op(self, input_tensor, is_training):
        self.n_input_chns = input_tensor.get_shape()[-1]
        output_tensor = input_tensor
        for (i, k) in enumerate(self.kernels):
            # create parameterised layers
            bn_op = BNLayer(regularizer=self.w_regularizer,
                            name='bn_{}'.format(i))
            acti_op = ActiLayer(func=self.acti_type,
                                regularizer=self.w_regularizer,
                                name='acti_{}'.format(i))
            conv_op = ConvLayer(n_output_chns=self.n_output_chns,
                                kernel_size=k,
                                stride=1,
                                w_initializer=self.w_initializer,
                                w_regularizer=self.w_regularizer,
                                name='conv_{}'.format(i))

            # connect layers
            output_tensor = bn_op(output_tensor, is_training)
            output_tensor = acti_op(output_tensor)
            output_tensor = conv_op(output_tensor)

        # make residual connections
        if self.with_res:
            merge_connections = ElementwiseSumLayer()
            output_tensor = merge_connections(output_tensor, input_tensor)
        return output_tensor
