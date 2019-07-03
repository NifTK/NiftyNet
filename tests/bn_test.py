# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.layer.bn import BNLayer
from niftynet.layer.bn import InstanceNormLayer
from tests.niftynet_testcase import NiftyNetTestCase


class BNTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def test_3d_bn_shape(self):
        x = self.get_3d_input()
        bn_layer = BNLayer()
        print(bn_layer)
        out_bn = bn_layer(x, is_training=True)
        print(bn_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            out = sess.run(out_bn)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

    def test_3d_instnorm_shape(self):
        x = self.get_3d_input()
        instnorm_layer = InstanceNormLayer()
        print(instnorm_layer)
        out_inst = instnorm_layer(x)
        print(instnorm_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            out = sess.run(out_inst)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

    def test_3d_bn_reg_shape(self):
        x = self.get_3d_input()
        bn_layer = BNLayer(regularizer=regularizers.l2_regularizer(0.5))
        out_bn = bn_layer(x, is_training=True)
        test_bn = bn_layer(x, is_training=False)
        print(bn_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            out = sess.run(out_bn)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

            out = sess.run(test_bn)
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

    def test_2d_bn_shape(self):
        x = self.get_2d_input()
        bn_layer = BNLayer()
        out_bn = bn_layer(x, is_training=True)
        print(bn_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            out = sess.run(out_bn)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

    def test_2d_instnorm_shape(self):
        x = self.get_2d_input()
        instnorm_layer = InstanceNormLayer()
        out_inst = instnorm_layer(x)
        print(instnorm_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            out = sess.run(out_inst)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

    def test_2d_bn_reg_shape(self):
        x = self.get_2d_input()
        bn_layer = BNLayer(regularizer=regularizers.l2_regularizer(0.5))
        out_bn = bn_layer(x, is_training=True)
        test_bn = bn_layer(x, is_training=False)
        print(bn_layer)
        reg_loss = tf.add_n(bn_layer.regularizer_loss())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            out = sess.run(out_bn)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

            out = sess.run(test_bn)
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

            out = sess.run(reg_loss)
            self.assertAlmostEqual(out, 2.0)


if __name__ == "__main__":
    tf.test.main()
