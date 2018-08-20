from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.layer.convolution import ConvLayer
from niftynet.layer.convolution import ConvolutionalLayer


class ConvTest(tf.test.TestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x_3d = tf.ones(input_shape)
        return x_3d

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x_2d = tf.ones(input_shape)
        return x_2d

    def _test_conv_output_shape(self,
                                rank,
                                param_dict,
                                output_shape):
        if rank == 2:
            input_data = self.get_2d_input()
        elif rank == 3:
            input_data = self.get_3d_input()

        conv_layer = ConvLayer(**param_dict)
        output_data = conv_layer(input_data)
        print(conv_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_value = sess.run(output_data)
            self.assertAllClose(output_shape, output_value.shape)

    def _test_conv_layer_output_shape(self,
                                      rank,
                                      param_dict,
                                      output_shape,
                                      is_training=None,
                                      dropout_prob=None):
        if rank == 2:
            input_data = self.get_2d_input()
        elif rank == 3:
            input_data = self.get_3d_input()

        conv_layer = ConvolutionalLayer(**param_dict)
        output_data = conv_layer(input_data,
                                 is_training=is_training,
                                 keep_prob=dropout_prob)
        print(conv_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_value = sess.run(output_data)
            self.assertAllClose(output_shape, output_value.shape)

    # 3d tests
    def test_3d_conv_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 16, 10))

    def test_3d_conv_full_kernel_size(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 3, 1],
                       'stride': 1}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 16, 10))

    def test_3d_conv_full_strides(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 3, 1],
                       'stride': [1, 1, 2]}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 8, 10))

    def test_3d_anisotropic_conv(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 2, 1],
                       'stride': [1, 1, 2]}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 8, 10))

    def test_3d_conv_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 1, 3],
                       'stride': [1, 1, 2],
                       'with_bias': True}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 8, 10))

    def test_conv_3d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 1, 3],
                       'stride': [2, 2, 2],
                       'with_bias': True,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 8, 8, 8, 10))

    def test_3d_convlayer_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 16, 10))

    def test_3d_convlayer_dilation_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1,
                       'dilation': [1, 2, 1]}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 16, 10))

    def test_3d_convlayer_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1,
                       'with_bias': True,
                       'with_bn': False}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 16, 10))

    def test_convlayer_3d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1,
                       'with_bias': True,
                       'with_bn': False,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 16, 10))

    def test_convlayer_3d_bn_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 1, 2],
                       'stride': 1,
                       'with_bias': False,
                       'with_bn': True,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 16, 10),
                                           is_training=True)

    def test_convlayer_3d_bn_reg_prelu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 1, 2],
                       'stride': [1, 1, 2],
                       'with_bias': False,
                       'with_bn': True,
                       'acti_func': 'prelu',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 8, 10),
                                           is_training=True)

    def test_convlayer_3d_relu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 1, 2],
                       'stride': [1, 2, 2],
                       'with_bias': False,
                       'with_bn': True,
                       'acti_func': 'relu',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 8, 8, 10),
                                           is_training=True)

    def test_convlayer_3d_bn_reg_dropout_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 1, 2],
                       'stride': [1, 2, 2],
                       'with_bias': False,
                       'with_bn': True,
                       'acti_func': 'prelu'}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 8, 8, 10),
                                           is_training=True,
                                           dropout_prob=0.4)

    def test_convlayer_3d_bn_reg_dropout_valid_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 3, 2],
                       'stride': [2, 2, 3],
                       'with_bias': False,
                       'with_bn': True,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'acti_func': 'prelu',
                       'padding': 'VALID'}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 6, 7, 5, 10),
                                           is_training=True,
                                           dropout_prob=0.4)

    def test_convlayer_3d_group_reg_dropout_valid_shape(self):
        input_param = {'n_output_chns': 8,
                       'kernel_size': [5, 3, 2],
                       'stride': [2, 2, 3],
                       'with_bias': False,
                       'with_bn': False,
                       'group_size': 4,
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 8, 8, 6, 8),
                                           is_training=True,
                                           dropout_prob=0.4)

    # 2d tests
    def test_2d_conv_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 3],
                       'stride': [2, 2]}
        self._test_conv_output_shape(rank=2,
                                     param_dict=input_param,
                                     output_shape=(2, 8, 8, 10))

    def test_2d_conv_dilation_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 3],
                       'stride': [1, 1],
                       'dilation': [2, 1]}
        self._test_conv_output_shape(rank=2,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 10))

    def test_2d_conv_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 3],
                       'stride': [1, 2],
                       'with_bias': True}
        self._test_conv_output_shape(rank=2,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 8, 10))

    def test_conv_2d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 5,
                       'stride': 1,
                       'with_bias': True,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_output_shape(rank=2,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 10))

    def test_2d_convlayer_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 2,
                       'stride': 1,
                       'with_bias': True}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 10),
                                           is_training=True)

    def test_2d_convlayer_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 2,
                       'stride': [2, 1],
                       'with_bias': True,
                       'with_bn': False}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 8, 16, 10))

    def test_convlayer_2d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 5],
                       'stride': [2, 1],
                       'with_bias': True,
                       'with_bn': False,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 8, 16, 10))

    def test_convlayer_2d_bn_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 5],
                       'stride': [2, 1],
                       'with_bias': False,
                       'with_bn': True,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 8, 16, 10),
                                           is_training=True)

    def test_convlayer_2d_bn_reg_prelu_2_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': [2, 1],
                       'with_bias': False,
                       'with_bn': True,
                       'acti_func': 'prelu'}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 8, 16, 10),
                                           is_training=True)

    def test_convlayer_2d_relu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': [3, 1],
                       'with_bias': False,
                       'with_bn': True,
                       'acti_func': 'relu'}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 6, 16, 10),
                                           is_training=True)

    def test_convlayer_2d_bn_reg_prelu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1,
                       'with_bias': False,
                       'with_bn': True,
                       'acti_func': 'prelu',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 10),
                                           is_training=True)

    def test_convlayer_2d_bn_reg_valid_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 2],
                       'stride': [2, 3],
                       'with_bias': False,
                       'with_bn': True,
                       'acti_func': 'prelu',
                       'padding': 'VALID',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 7, 5, 10),
                                           is_training=True)


if __name__ == "__main__":
    tf.test.main()
