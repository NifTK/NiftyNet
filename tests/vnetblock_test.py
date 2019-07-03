from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.vnet import VNetBlock
from tests.niftynet_testcase import NiftyNetTestCase

class VNetBlockTest(NiftyNetTestCase):
    def get_2d_data(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_3d_data(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def test_3d_shape(self):
        x = self.get_3d_data()
        vnet_block_op = VNetBlock('DOWNSAMPLE', 2, 16, 8)
        out_1, out_2 = vnet_block_op(x, x)
        print(vnet_block_op)

        vnet_block_op = VNetBlock('UPSAMPLE', 2, 16, 8)
        out_3, out_4 = vnet_block_op(x, x)
        print(vnet_block_op)

        vnet_block_op = VNetBlock('SAME', 2, 16, 8)
        out_5, out_6 = vnet_block_op(x, x)
        print(vnet_block_op)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            self.assertAllClose((2, 16, 16, 16, 16), out_1.shape)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 8, 8, 8, 8), out_2.shape)
            out_3 = sess.run(out_3)
            self.assertAllClose((2, 16, 16, 16, 16), out_3.shape)
            out_4 = sess.run(out_4)
            self.assertAllClose((2, 32, 32, 32, 8), out_4.shape)
            out_5 = sess.run(out_5)
            self.assertAllClose((2, 16, 16, 16, 16), out_5.shape)
            out_6 = sess.run(out_6)
            self.assertAllClose((2, 16, 16, 16, 8), out_6.shape)

    def test_2d_shape(self):
        x = self.get_2d_data()
        vnet_block_op = VNetBlock('DOWNSAMPLE', 2, 16, 8)
        out_1, out_2 = vnet_block_op(x, x)
        print(vnet_block_op)

        vnet_block_op = VNetBlock('UPSAMPLE', 2, 16, 8)
        out_3, out_4 = vnet_block_op(x, x)
        print(vnet_block_op)

        vnet_block_op = VNetBlock('SAME', 2, 16, 8)
        out_5, out_6 = vnet_block_op(x, x)
        print(vnet_block_op)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            self.assertAllClose((2, 16, 16, 16), out_1.shape)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 8, 8, 8), out_2.shape)
            out_3 = sess.run(out_3)
            self.assertAllClose((2, 16, 16, 16), out_3.shape)
            out_4 = sess.run(out_4)
            self.assertAllClose((2, 32, 32, 8), out_4.shape)
            out_5 = sess.run(out_5)
            self.assertAllClose((2, 16, 16, 16), out_5.shape)
            out_6 = sess.run(out_6)
            self.assertAllClose((2, 16, 16, 8), out_6.shape)

    def test_3d_reg_shape(self):
        x = self.get_3d_data()
        vnet_block_op = VNetBlock('DOWNSAMPLE', 2, 16, 8,
                                  w_regularizer=regularizers.l2_regularizer(
                                      0.2))
        out_1, out_2 = vnet_block_op(x, x)
        print(vnet_block_op)

        vnet_block_op = VNetBlock('UPSAMPLE', 2, 16, 8,
                                  w_regularizer=regularizers.l2_regularizer(
                                      0.2))
        out_3, out_4 = vnet_block_op(x, x)
        print(vnet_block_op)

        vnet_block_op = VNetBlock('SAME', 2, 16, 8,
                                  w_regularizer=regularizers.l2_regularizer(
                                      0.2))
        out_5, out_6 = vnet_block_op(x, x)
        print(vnet_block_op)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            self.assertAllClose((2, 16, 16, 16, 16), out_1.shape)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 8, 8, 8, 8), out_2.shape)
            out_3 = sess.run(out_3)
            self.assertAllClose((2, 16, 16, 16, 16), out_3.shape)
            out_4 = sess.run(out_4)
            self.assertAllClose((2, 32, 32, 32, 8), out_4.shape)
            out_5 = sess.run(out_5)
            self.assertAllClose((2, 16, 16, 16, 16), out_5.shape)
            out_6 = sess.run(out_6)
            self.assertAllClose((2, 16, 16, 16, 8), out_6.shape)

    def test_2d_reg_shape(self):
        x = self.get_2d_data()
        vnet_block_op = VNetBlock('DOWNSAMPLE', 2, 16, 8,
                                  w_regularizer=regularizers.l2_regularizer(
                                      0.2))
        out_1, out_2 = vnet_block_op(x, x)
        print(vnet_block_op)

        vnet_block_op = VNetBlock('UPSAMPLE', 2, 16, 8,
                                  w_regularizer=regularizers.l2_regularizer(
                                      0.2))
        out_3, out_4 = vnet_block_op(x, x)
        print(vnet_block_op)

        vnet_block_op = VNetBlock('SAME', 2, 16, 8,
                                  w_regularizer=regularizers.l2_regularizer(
                                      0.2))
        out_5, out_6 = vnet_block_op(x, x)
        print(vnet_block_op)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            self.assertAllClose((2, 16, 16, 16), out_1.shape)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 8, 8, 8), out_2.shape)
            out_3 = sess.run(out_3)
            self.assertAllClose((2, 16, 16, 16), out_3.shape)
            out_4 = sess.run(out_4)
            self.assertAllClose((2, 32, 32, 8), out_4.shape)
            out_5 = sess.run(out_5)
            self.assertAllClose((2, 16, 16, 16), out_5.shape)
            out_6 = sess.run(out_6)
            self.assertAllClose((2, 16, 16, 8), out_6.shape)


if __name__ == "__main__":
    tf.test.main()
