# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.layer.activation import ActiLayer
from tests.niftynet_testcase import NiftyNetTestCase

class ActivationTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def run_test(self, is_3d, type_str, expected_shape):
        if is_3d:
            x = self.get_3d_input()
        else:
            x = self.get_2d_input()
        activation_layer = ActiLayer(func=type_str)
        out_acti = activation_layer(x)
        print(activation_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_acti)
            self.assertAllClose(out.shape, expected_shape)

    # 3d test
    def test_3d_relu_shape(self):
        self.run_test(True, 'relu', (2, 16, 16, 16, 8))

    def test_3d_relu6_shape(self):
        self.run_test(True, 'relu6', (2, 16, 16, 16, 8))

    def test_3d_elu_shape(self):
        self.run_test(True, 'elu', (2, 16, 16, 16, 8))

    def test_3d_selu_shape(self):
        self.run_test(True, 'selu', (2, 16, 16, 16, 8))

    def test_3d_softplus_shape(self):
        self.run_test(True, 'softplus', (2, 16, 16, 16, 8))

    def test_3d_softsign_shape(self):
        self.run_test(True, 'softsign', (2, 16, 16, 16, 8))

    def test_3d_sigmoid_shape(self):
        self.run_test(True, 'sigmoid', (2, 16, 16, 16, 8))

    def test_3d_tanh_shape(self):
        self.run_test(True, 'tanh', (2, 16, 16, 16, 8))

    def test_3d_prelu_shape(self):
        self.run_test(True, 'prelu', (2, 16, 16, 16, 8))

    def test_3d_prelu_reg_shape(self):
        x = self.get_3d_input()
        prelu_layer = ActiLayer(func='prelu',
                                regularizer=regularizers.l2_regularizer(0.5),
                                name='regularized')
        out_prelu = prelu_layer(x)
        print(prelu_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_prelu)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_dropout_shape(self):
        x = self.get_3d_input()
        dropout_layer = ActiLayer(func='dropout')
        out_dropout = dropout_layer(x, keep_prob=0.8)
        print(dropout_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_dropout)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

            # self.assertAllClose(input_shape, out.shape)
            # self.assertAllClose(np.zeros(input_shape), out)

    # 2d test
    def test_2d_relu_shape(self):
        self.run_test(False, 'relu', (2, 16, 16, 8))

    def test_2d_relu6_shape(self):
        self.run_test(False, 'relu6', (2, 16, 16, 8))

    def test_2d_elu_shape(self):
        self.run_test(False, 'elu', (2, 16, 16, 8))

    def test_2d_softplus_shape(self):
        self.run_test(False, 'softplus', (2, 16, 16, 8))

    def test_2d_softsign_shape(self):
        self.run_test(False, 'softsign', (2, 16, 16, 8))

    def test_2d_sigmoid_shape(self):
        self.run_test(False, 'sigmoid', (2, 16, 16, 8))

    def test_2d_tanh_shape(self):
        self.run_test(False, 'tanh', (2, 16, 16, 8))

    def test_2d_prelu_shape(self):
        self.run_test(False, 'prelu', (2, 16, 16, 8))

    def test_2d_selu_shape(self):
        self.run_test(False, 'selu', (2, 16, 16, 8))

    def test_2d_prelu_reg_shape(self):
        x = self.get_2d_input()
        prelu_layer = ActiLayer(func='prelu',
                                regularizer=regularizers.l2_regularizer(0.5),
                                name='regularized')
        out_prelu = prelu_layer(x)
        print(prelu_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_prelu)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_dropout_shape(self):
        x = self.get_2d_input()
        dropout_layer = ActiLayer(func='dropout')
        out_dropout = dropout_layer(x, keep_prob=0.8)
        print(dropout_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_dropout)
            self.assertAllClose((2, 16, 16, 8), out.shape)


if __name__ == "__main__":
    tf.test.main()
