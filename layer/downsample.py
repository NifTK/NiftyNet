import tensorflow as tf

from base import Layer


SUPPORTED_OP = set(['AVG', 'MAX'])
SUPPORTED_PADDING = set(['SAME', 'VALID'])


class DownSampleLayer(Layer):
    def __init__(self,
                 func,
                 kernel_size,
                 stride,
                 padding='SAME',
                 name='pooling'):
        self.func = func.upper()
        self.padding = padding.upper()
        assert(self.func in SUPPORTED_OP)
        assert(self.padding in SUPPORTED_PADDING)

        self.kernel_size = kernel_size
        self.stride = stride

        self.layer_name = '{}_{}'.format(self.func.lower(), name)
        super(DownSampleLayer, self).__init__(name=self.layer_name)

    def layer_op(self, input_tensor):
        spatial_rank = max(input_tensor.get_shape().ndims - 2, 0)
        output_tensor = tf.nn.pool(
                input=input_tensor,
                window_shape=[self.kernel_size] * spatial_rank,
                pooling_type=self.func,
                padding=self.padding,
                dilation_rate=[1] * spatial_rank,
                strides=[self.stride] * spatial_rank,
                name=self.layer_name)
        return output_tensor
