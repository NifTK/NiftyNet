# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.layer.gn import GNLayer
from tests.niftynet_testcase import NiftyNetTestCase

class GNTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def test_3d_gn_shape(self):
        x = self.get_3d_input()
        gn_layer = GNLayer(4)
        print(gn_layer)
        out_gn = gn_layer(x)
        print(gn_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            out = sess.run(out_gn)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

    def test_3d_gn_reg_shape(self):
        x = self.get_3d_input()
        gn_layer = GNLayer(4, regularizer=regularizers.l2_regularizer(0.5))
        out_gn = gn_layer(x)
        test_gn = gn_layer(x)
        print(gn_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            out = sess.run(out_gn)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

            out = sess.run(test_gn)
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

    def test_2d_gn_shape(self):
        x = self.get_2d_input()
        gn_layer = GNLayer(4)
        out_gn = gn_layer(x)
        print(gn_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            out = sess.run(out_gn)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

    def test_2d_gn_reg_shape(self):
        x = self.get_2d_input()
        gn_layer = GNLayer(4, regularizer=regularizers.l2_regularizer(0.5))
        out_gn = gn_layer(x)
        test_gn = gn_layer(x)
        print(gn_layer)
        reg_loss = tf.add_n(gn_layer.regularizer_loss())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            out = sess.run(out_gn)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

            out = sess.run(test_gn)
            self.assertAllClose(x_shape, out.shape)
            # self.assertAllClose(np.zeros(x_shape), out)

            out = sess.run(reg_loss)
            self.assertAlmostEqual(out, 2.0)


if __name__ == "__main__":
    tf.test.main()
