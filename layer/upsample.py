import numpy as np
import tensorflow as tf

from .base import Layer
from .deconvolution import DeconvLayer
from . import layer_util


SUPPORTED_OP = set(['REPLICATE', 'CHANNELWISE_DECONV'])


class UpSampleLayer(Layer):
    """
    This class defines channel-wise upsampling operations.
    Different from `DeconvLayer`, the elements are not mixed in the channel dim.
    'REPLICATE' mode replicates each spatial_dim into spatial_dim*kernel_size
    'CHANNELWISE_DECONV' model makes a projection using a kernel.
    e.g., With 2D input (without loss of generality), given input [N, X, Y, C],
    the output is [N, X*kernel_size, Y*kernel_size, C].
    """

    def __init__(self,
                 func,
                 kernel_size,
                 stride,
                 w_initializer=None,
                 w_regularizer=None,
                 with_bias=False,
                 b_initializer=None,
                 b_regularizer=None,
                 name='upsample'):
        self.func = func.upper()
        assert self.func in SUPPORTED_OP
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_name = '{}_{}'.format(self.func.lower(), name)
        super(UpSampleLayer, self).__init__(name=self.layer_name)

        self.with_bias = with_bias
        self.w_initializer = w_initializer
        self.w_regularizer = w_regularizer
        self.b_initializer = b_initializer
        self.b_regularizer = b_regularizer

    def layer_op(self, input_tensor):
        spatial_rank = layer_util.infer_spatial_rank(input_tensor)
        if self.func == 'REPLICATE':
            if self.kernel_size != self.stride:
                raise ValueError(
                    "`kernel_size` != `stride` currently not"
                    "supported in upsampling layer. Please"
                    "consider using `CHANNELWISE_DECONV` operation.")
            # simply replicate input values to
            # local regions of (kernel_size ** spatial_rank) element
            repmat = np.hstack((self.kernel_size**spatial_rank,
                                [1] * spatial_rank, 1)).flatten()
            output_tensor = tf.tile(input=input_tensor, multiples=repmat)
            output_tensor = tf.batch_to_space_nd(
                output_tensor,
                [self.kernel_size] * spatial_rank,
                [[0, 0]] * spatial_rank)

        elif self.func == 'CHANNELWISE_DECONV':
            output_tensor = [tf.expand_dims(x, -1)
                             for x in tf.unstack(input_tensor, axis=-1)]
            output_tensor = [DeconvLayer(n_output_chns=1,
                                         kernel_size=self.kernel_size,
                                         stride=self.stride,
                                         padding='SAME',
                                         with_bias=self.with_bias,
                                         w_initializer=self.w_initializer,
                                         w_regularizer=self.w_regularizer,
                                         b_initializer=self.b_initializer,
                                         b_regularizer=self.b_regularizer,
                                         name='deconv_{}'.format(i))(x)
                             for (i, x) in enumerate(output_tensor)]
            output_tensor = tf.concat(output_tensor, axis=-1)

        return output_tensor
