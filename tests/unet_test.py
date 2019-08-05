from __future__ import absolute_import, print_function

import unittest

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.unet import UNet3D
from tests.niftynet_testcase import NiftyNetTestCase

@unittest.skip('Test currently disabled')
class UNet3DTest(NiftyNetTestCase):
    def test_3d_shape(self):
        input_shape = (2, 96, 96, 96, 1)
        x = tf.ones(input_shape)

        unet_instance = UNet3D(num_classes=160)
        out = unet_instance(x, is_training=True)
        print(unet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 8, 8, 8, 160), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 96, 96, 1)
        x = tf.ones(input_shape)

        unet_instance = UNet3D(num_classes=160)
        out = unet_instance(x, is_training=True)
        print(unet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 8, 8, 160), out.shape)

    def test_3d_reg_shape(self):
        input_shape = (2, 96, 96, 96, 1)
        x = tf.ones(input_shape)

        unet_instance = UNet3D(num_classes=160,
                               w_regularizer=regularizers.l2_regularizer(0.4))
        out = unet_instance(x, is_training=True)
        print(unet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 8, 8, 8, 160), out.shape)

    def test_2d_reg_shape(self):
        input_shape = (2, 96, 96, 1)
        x = tf.ones(input_shape)

        unet_instance = UNet3D(num_classes=160,
                               w_regularizer=regularizers.l2_regularizer(0.4))
        out = unet_instance(x, is_training=True)
        print(unet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 8, 8, 160), out.shape)


if __name__ == "__main__":
    tf.test.main()
