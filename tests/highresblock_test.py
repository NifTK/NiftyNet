from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.highres3dnet import HighResBlock
from tests.niftynet_testcase import NiftyNetTestCase


class HighResBlockTest(NiftyNetTestCase):
    def test_3d_increase_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)

        highres_layer = HighResBlock(n_output_chns=16,
                                     kernels=(3, 3),
                                     with_res=True)
        out = highres_layer(x, is_training=True)
        print(highres_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 16, 16, 16, 16), out.shape)

    def test_3d_same_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)

        highres_layer = HighResBlock(n_output_chns=8,
                                     kernels=(3, 3),
                                     with_res=True)
        out = highres_layer(x, is_training=True)
        print(highres_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_reduce_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)

        highres_layer = HighResBlock(n_output_chns=4,
                                     kernels=(3, 3),
                                     with_res=True)
        out = highres_layer(x, is_training=True)
        print(highres_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 16, 16, 16, 4), out.shape)

    def test_3d_reg_increase_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)

        highres_layer = HighResBlock(
            n_output_chns=16,
            kernels=(3, 3),
            with_res=True,
            w_regularizer=regularizers.l2_regularizer(0.3))
        out = highres_layer(x, is_training=True)
        print(highres_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 16, 16, 16, 16), out.shape)

    def test_3d_reg_same_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)

        highres_layer = HighResBlock(
            n_output_chns=8,
            kernels=(3, 3),
            with_res=True,
            w_regularizer=regularizers.l2_regularizer(0.3))

        out = highres_layer(x, is_training=True)
        print(highres_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_reg_reduce_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)

        highres_layer = HighResBlock(
            n_output_chns=4,
            kernels=(3, 3),
            with_res=True,
            w_regularizer=regularizers.l2_regularizer(0.3))
        out = highres_layer(x, is_training=True)
        print(highres_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 16, 16, 16, 4), out.shape)

    def test_2d_increase_shape(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)

        highres_layer = HighResBlock(n_output_chns=16,
                                     kernels=(3, 3),
                                     with_res=True)
        out = highres_layer(x, is_training=True)
        print(highres_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 16, 16, 16), out.shape)

    def test_2d_same_shape(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)

        highres_layer = HighResBlock(n_output_chns=8,
                                     kernels=(3, 3),
                                     with_res=True)
        out = highres_layer(x, is_training=True)
        print(highres_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_reduce_shape(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)

        highres_layer = HighResBlock(n_output_chns=4,
                                     kernels=(3, 3),
                                     with_res=True)
        out = highres_layer(x, is_training=True)
        print(highres_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 16, 16, 4), out.shape)


if __name__ == "__main__":
    tf.test.main()
