from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.layer.deconvolution import DeconvLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from tests.niftynet_testcase import NiftyNetTestCase


class DeconvTest(NiftyNetTestCase):
    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x_2d = tf.ones(input_shape)
        return x_2d

    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x_3d = tf.ones(input_shape)
        return x_3d

    def _test_deconv_output_shape(self,
                                  rank,
                                  param_dict,
                                  output_shape):
        if rank == 2:
            input_data = self.get_2d_input()
        elif rank == 3:
            input_data = self.get_3d_input()

        deconv_layer = DeconvLayer(**param_dict)
        output_data = deconv_layer(input_data)
        print(deconv_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_value = sess.run(output_data)
            self.assertAllClose(output_shape, output_value.shape)

    def _test_deconv_layer_output_shape(self,
                                        rank,
                                        param_dict,
                                        output_shape,
                                        is_training=None,
                                        dropout_prob=None):
        if rank == 2:
            input_data = self.get_2d_input()
        elif rank == 3:
            input_data = self.get_3d_input()

        deconv_layer = DeconvolutionalLayer(**param_dict)
        output_data = deconv_layer(input_data,
                                   is_training=is_training,
                                   keep_prob=dropout_prob)
        print(deconv_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_value = sess.run(output_data)
            self.assertAllClose(output_shape, output_value.shape)

    def test_3d_deconv_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 3, 1],
                       'stride': 2}
        self._test_deconv_output_shape(rank=3,
                                       param_dict=input_param,
                                       output_shape=(2, 32, 32, 32, 10))

    def test_3d_deconv_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 2,
                       'with_bias': True}
        self._test_deconv_output_shape(rank=3,
                                       param_dict=input_param,
                                       output_shape=(2, 32, 32, 32, 10))

    def test_deconv_3d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': [2, 2, 1],
                       'with_bias': True,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_output_shape(rank=3,
                                       param_dict=input_param,
                                       output_shape=(2, 32, 32, 16, 10))

    def test_3d_deconvlayer_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1}
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 16, 16, 10),
                                             is_training=True)

    def test_3d_deconvlayer_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 2,
                       'with_bias': True,
                       'feature_normalization': None}
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 32, 32, 32, 10),
                                             is_training=True)
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 32, 32, 32, 10),
                                             is_training=False)

    def test_deconvlayer_3d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1,
                       'with_bias': True,
                       'feature_normalization': None,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 16, 16, 10))
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 16, 16, 10))

    def test_deconvlayer_3d_bn_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 16, 16, 10),
                                             is_training=True)
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 16, 16, 10),
                                             is_training=False)

    def test_deconvlayer_3d_bn_reg_prelu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 5, 2],
                       'stride': [1, 1, 2],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'acti_func': 'prelu'}
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 16, 32, 10),
                                             is_training=True)
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 16, 32, 10),
                                             is_training=False)

    def test_deconvlayer_3d_relu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 5, 2],
                       'stride': [1, 1, 2],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'relu'}
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 16, 32, 10),
                                             is_training=True)
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 16, 32, 10),
                                             is_training=False)

    def test_deconvlayer_3d_bn_reg_dropout_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 5, 2],
                       'stride': [1, 2, 2],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 32, 32, 10),
                                             is_training=True)
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 32, 32, 10),
                                             is_training=False)

    def test_deconvlayer_3d_bn_reg_dropout_valid_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 5, 2],
                       'stride': [1, 2, 1],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 32, 16, 10),
                                             is_training=True,
                                             dropout_prob=0.4)
        self._test_deconv_layer_output_shape(rank=3,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 32, 16, 10),
                                             is_training=False,
                                             dropout_prob=1.0)

    def test_2d_deconv_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 1],
                       'stride': [2, 1]}
        self._test_deconv_output_shape(rank=2,
                                       param_dict=input_param,
                                       output_shape=(2, 32, 16, 10))

    def test_2d_deconv_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 1],
                       'stride': [2, 1],
                       'with_bias': True}
        self._test_deconv_output_shape(rank=2,
                                       param_dict=input_param,
                                       output_shape=(2, 32, 16, 10))

    def test_deconv_2d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 1],
                       'stride': [2, 1],
                       'with_bias': True,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_output_shape(rank=2,
                                       param_dict=input_param,
                                       output_shape=(2, 32, 16, 10))

    def test_2d_deconvlayer_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 1],
                       'stride': [2, 1]}
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 32, 16, 10),
                                             is_training=True)
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 32, 16, 10),
                                             is_training=False)

    def test_2d_deconvlayer_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 1],
                       'stride': [2, 1],
                       'with_bias': True,
                       'feature_normalization': None}
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 32, 16, 10),
                                             is_training=True)
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 32, 16, 10),
                                             is_training=False)

    def test_deconvlayer_2d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 1],
                       'stride': [2, 3],
                       'with_bias': True,
                       'feature_normalization': None,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 32, 48, 10),
                                             is_training=True)
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 32, 48, 10),
                                             is_training=False)

    def test_deconvlayer_2d_bn_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 1],
                       'stride': [1, 3],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 48, 10),
                                             is_training=True)
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 48, 10),
                                             is_training=False)

    def test_deconvlayer_2d_bn_reg_prelu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [4, 1],
                       'stride': [1, 3],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 48, 10),
                                             is_training=True)
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 48, 10),
                                             is_training=False)

    def test_deconvlayer_2d_relu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [4, 1],
                       'stride': [1, 3],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'relu',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 48, 10),
                                             is_training=True)
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 48, 10),
                                             is_training=False)

    def test_deconvlayer_2d_bn_reg_dropout_prelu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [4, 1],
                       'stride': [1, 3],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 48, 10),
                                             is_training=True,
                                             dropout_prob=0.4)
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 16, 48, 10),
                                             is_training=False,
                                             dropout_prob=1.0)

    def test_deconvlayer_2d_bn_reg_valid_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [4, 3],
                       'stride': [1, 2],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu',
                       'padding': 'VALID',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 19, 33, 10),
                                             is_training=True,
                                             dropout_prob=0.4)
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 19, 33, 10),
                                             is_training=False,
                                             dropout_prob=1.0)

    def test_deconvlayer_2d_group_reg_valid_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [4, 3],
                       'stride': [1, 2],
                       'with_bias': False,
                       'feature_normalization': 'group',
                       'group_size': 5,
                       'acti_func': 'prelu',
                       'padding': 'VALID',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 19, 33, 10),
                                             is_training=True,
                                             dropout_prob=0.4)
        self._test_deconv_layer_output_shape(rank=2,
                                             param_dict=input_param,
                                             output_shape=(2, 19, 33, 10),
                                             is_training=False,
                                             dropout_prob=1.0)


if __name__ == "__main__":
    tf.test.main()
