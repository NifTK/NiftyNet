import numpy as np
import tensorflow as tf

from base import Layer
from bn import BNLayer


SUPPORTED_OP = set([tf.nn.conv1d, tf.nn.conv2d, tf.nn.conv3d])
SUPPORTED_PADDING = set(['SAME', 'VALID'])

def default_w_initializer(kernel_shape):
    stddev = np.sqrt(1.3 * 2.0 / \
            (np.prod(kernel_shape[:-2]) * kernel_shape[-1]))
    return tf.truncated_normal_initializer(
            mean=0.0, stddev=stddev, dtype=tf.float32)

def default_b_initializer():
    return tf.zeros_initializer


class ConvLayer(Layer):
    def __init__(self,
                 conv_op,
                 n_output_chns,
                 kernel_size=[3, 3, 3],
                 strides=[1, 1, 1, 1, 1],
                 w_initializer=None,
                 with_bias=False,
                 b_initializer=None,
                 with_bn=False,
                 with_acti=False,
                 padding='SAME',
                 name='ConvLayer'):

        assert(conv_op in SUPPORTED_OP)
        assert(padding in SUPPORTED_PADDING)

        self.layer_name = '{}_{}'.format(name, conv_op.__name__)
        super(ConvLayer, self).__init__(name=self.layer_name)

        self.n_output_chns= n_output_chns
        self.conv_op = conv_op
        self.kernel_size = np.asarray(kernel_size).flatten()
        self.strides = np.asarray(strides).flatten()
        self.w_initializer = w_initializer
        self.with_bias = with_bias
        self.b_initializer = b_initializer
        self.with_bn = with_bn
        self.with_acti = with_acti
        self.padding = padding
        self._w = None
        self._b = None

    def layer_op(self, input_tensor, **kwargs):
        input_shape = input_tensor.get_shape().as_list()
        n_input_chns = input_shape[-1]
        w_full_size = np.hstack((
            self.kernel_size, n_input_chns, self.n_output_chns)).flatten()

        if self.w_initializer is None:
            self.w_initializer = default_w_initializer(w_full_size)
        self._w = tf.get_variable(
                'w', shape=w_full_size.tolist(), initializer=self.w_initializer)
        output_tensor = self.conv_op(
                input=input_tensor,
                filter=self._w,
                strides=self.strides.tolist(),
                padding=self.padding)

        if (self.with_bias) and (not self.with_bn):
            if self.b_initializer is None:
                self.b_initializer = default_b_initializer()
            self._b = tf.get_variable(
                    'b', shape=(self.n_output_chns),
                    initializer=self.b_initializer)
            output_tensor = tf.nn.bias_add(output_tensor, self._b)

        if self.with_bn:
            bn_layer = BNLayer()
            output_tensor = bn_layer(output_tensor, **kwargs)

        if self.with_acti:
            pass
        return output_tensor
