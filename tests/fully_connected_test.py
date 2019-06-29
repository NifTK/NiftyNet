from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.layer.fully_connected import FCLayer
from niftynet.layer.fully_connected import FullyConnectedLayer
from tests.niftynet_testcase import NiftyNetTestCase

class FCTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (2, 10, 10, 5, 8)
        x_3d = tf.ones(input_shape)
        return x_3d

    def get_2d_input(self):
        input_shape = (2, 8, 4, 8)
        x_2d = tf.ones(input_shape)
        return x_2d

    def _test_fc_output_shape(self,
                              rank,
                              param_dict,
                              output_shape):
        if rank == 2:
            input_data = self.get_2d_input()
        elif rank == 3:
            input_data = self.get_3d_input()

        fc_layer = FCLayer(**param_dict)
        output_data = fc_layer(input_data)
        print(fc_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_value = sess.run(output_data)
            self.assertAllClose(output_shape, output_value.shape)

    def _test_fc_layer_output_shape(self,
                                    rank,
                                    param_dict,
                                    output_shape,
                                    is_training=None,
                                    dropout_prob=None):
        if rank == 2:
            input_data = self.get_2d_input()
        elif rank == 3:
            input_data = self.get_3d_input()

        fc_layer = FullyConnectedLayer(**param_dict)
        output_data = fc_layer(input_data,
                               is_training=is_training,
                               keep_prob=dropout_prob)
        print(fc_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_value = sess.run(output_data)
            self.assertAllClose(output_shape, output_value.shape)

    # 3d tests
    def test_3d_fc_default_shape(self):
        input_param = {'n_output_chns': 10}
        self._test_fc_output_shape(rank=3,
                                   param_dict=input_param,
                                   output_shape=(2, 10))

    def test_3d_fc_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': True}
        self._test_fc_output_shape(rank=3,
                                   param_dict=input_param,
                                   output_shape=(2, 10))

    def test_fc_3d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': True,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_fc_output_shape(rank=3,
                                   param_dict=input_param,
                                   output_shape=(2, 10))

    def test_3d_fclayer_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': True,
                       'feature_normalization': None}
        self._test_fc_layer_output_shape(rank=3,
                                         param_dict=input_param,
                                         output_shape=(2, 10))

    def test_fclayer_3d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'feature_normalization': None,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_fc_layer_output_shape(rank=3,
                                         param_dict=input_param,
                                         output_shape=(2, 10))

    def test_fclayer_3d_bn_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_fc_layer_output_shape(rank=3,
                                         param_dict=input_param,
                                         output_shape=(2, 10),
                                         is_training=True)

    def test_fclayer_3d_bn_reg_prelu_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_fc_layer_output_shape(rank=3,
                                         param_dict=input_param,
                                         output_shape=(2, 10),
                                         is_training=True)

    def test_fclayer_3d_relu_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'relu',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_fc_layer_output_shape(rank=3,
                                         param_dict=input_param,
                                         output_shape=(2, 10),
                                         is_training=True)

    def test_fclayer_3d_bn_reg_dropout_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu'}
        self._test_fc_layer_output_shape(rank=3,
                                         param_dict=input_param,
                                         output_shape=(2, 10),
                                         is_training=True,
                                         dropout_prob=0.4)

    def test_fclayer_3d_bn_reg_dropout_valid_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'acti_func': 'prelu', }
        self._test_fc_layer_output_shape(rank=3,
                                         param_dict=input_param,
                                         output_shape=(2, 10),
                                         is_training=True,
                                         dropout_prob=0.4)

    # 2d tests
    def test_2d_fc_default_shape(self):
        input_param = {'n_output_chns': 10}
        self._test_fc_output_shape(rank=2,
                                   param_dict=input_param,
                                   output_shape=(2, 10))

    def test_2d_fc_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': True}
        self._test_fc_output_shape(rank=2,
                                   param_dict=input_param,
                                   output_shape=(2, 10))

    def test_fc_2d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': True,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_fc_output_shape(rank=2,
                                   param_dict=input_param,
                                   output_shape=(2, 10))

    def test_2d_fclayer_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': True}
        self._test_fc_layer_output_shape(rank=2,
                                         param_dict=input_param,
                                         output_shape=(2, 10),
                                         is_training=True)

    def test_2d_fclayer_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': True,
                       'feature_normalization': None}
        self._test_fc_layer_output_shape(rank=2,
                                         param_dict=input_param,
                                         output_shape=(2, 10))

    def test_fclayer_2d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': True,
                       'feature_normalization': None,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_fc_layer_output_shape(rank=2,
                                         param_dict=input_param,
                                         output_shape=(2, 10))

    def test_fclayer_2d_bn_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_fc_layer_output_shape(rank=2,
                                         param_dict=input_param,
                                         output_shape=(2, 10),
                                         is_training=True)

    def test_fclayer_2d_bn_reg_prelu_2_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu'}
        self._test_fc_layer_output_shape(rank=2,
                                         param_dict=input_param,
                                         output_shape=(2, 10),
                                         is_training=True)

    def test_fclayer_2d_relu_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'relu'}
        self._test_fc_layer_output_shape(rank=2,
                                         param_dict=input_param,
                                         output_shape=(2, 10),
                                         is_training=True)

    def test_fclayer_2d_bn_reg_prelu_shape(self):
        input_param = {'n_output_chns': 10,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_fc_layer_output_shape(rank=2,
                                         param_dict=input_param,
                                         output_shape=(2, 10),
                                         is_training=True)


if __name__ == "__main__":
    tf.test.main()
