from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.unet import UNetBlock
from tests.niftynet_testcase import NiftyNetTestCase

class UNetBlockTest(NiftyNetTestCase):
    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def test_2d_shape(self):
        x = self.get_2d_input()

        unet_block_op = UNetBlock(
            'DOWNSAMPLE', (32, 64), (3, 3), with_downsample_branch=True)
        out_1, out_2 = unet_block_op(x, is_training=True)
        print(unet_block_op)
        print(out_1)
        print(out_2)

        unet_block_op = UNetBlock(
            'UPSAMPLE', (32, 64), (3, 3), with_downsample_branch=False)
        out_3, _ = unet_block_op(x, is_training=True)
        print(unet_block_op)
        print(out_3)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            self.assertAllClose((2, 8, 8, 64), out_1.shape)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 16, 16, 64), out_2.shape)
            out_3 = sess.run(out_3)
            self.assertAllClose((2, 32, 32, 64), out_3.shape)

    def test_3d_shape(self):
        x = self.get_3d_input()

        unet_block_op = UNetBlock(
            'DOWNSAMPLE', (32, 64), (3, 3), with_downsample_branch=True)
        out_1, out_2 = unet_block_op(x, is_training=True)
        print(unet_block_op)
        print(out_1)
        print(out_2)

        unet_block_op = UNetBlock(
            'UPSAMPLE', (32, 64), (3, 3), with_downsample_branch=False)
        out_3, _ = unet_block_op(x, is_training=True)
        print(unet_block_op)
        print(out_3)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            self.assertAllClose((2, 8, 8, 8, 64), out_1.shape)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 16, 16, 16, 64), out_2.shape)
            out_3 = sess.run(out_3)
            self.assertAllClose((2, 32, 32, 32, 64), out_3.shape)

    def test_2d_reg_shape(self):
        x = self.get_2d_input()

        unet_block_op = UNetBlock(
            'DOWNSAMPLE', (32, 64), (3, 3), with_downsample_branch=True,
            w_regularizer=regularizers.l2_regularizer(0.3))
        out_1, out_2 = unet_block_op(x, is_training=True)
        print(unet_block_op)
        print(out_1)
        print(out_2)

        unet_block_op = UNetBlock(
            'UPSAMPLE', (32, 64), (3, 3), with_downsample_branch=False,
            w_regularizer=regularizers.l2_regularizer(0.3))
        out_3, _ = unet_block_op(x, is_training=True)
        print(unet_block_op)
        print(out_3)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            self.assertAllClose((2, 8, 8, 64), out_1.shape)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 16, 16, 64), out_2.shape)
            out_3 = sess.run(out_3)
            self.assertAllClose((2, 32, 32, 64), out_3.shape)

    def test_3d_reg_shape(self):
        x = self.get_3d_input()

        unet_block_op = UNetBlock(
            'DOWNSAMPLE', (32, 64), (3, 3), with_downsample_branch=True,
            w_regularizer=regularizers.l2_regularizer(0.3))
        out_1, out_2 = unet_block_op(x, is_training=True)
        print(unet_block_op)
        print(out_1)
        print(out_2)

        unet_block_op = UNetBlock(
            'UPSAMPLE', (32, 64), (3, 3), with_downsample_branch=False,
            w_regularizer=regularizers.l2_regularizer(0.3))
        out_3, _ = unet_block_op(x, is_training=True)
        print(unet_block_op)
        print(out_3)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            self.assertAllClose((2, 8, 8, 8, 64), out_1.shape)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 16, 16, 16, 64), out_2.shape)
            out_3 = sess.run(out_3)
            self.assertAllClose((2, 32, 32, 32, 64), out_3.shape)


if __name__ == "__main__":
    tf.test.main()
