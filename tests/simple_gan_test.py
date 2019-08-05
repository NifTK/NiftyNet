from __future__ import absolute_import, print_function

import unittest

import os
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.simple_gan import SimpleGAN
from tests.niftynet_testcase import NiftyNetTestCase

class SimpleGANTest(NiftyNetTestCase):
    def test_3d_reg_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        noise_shape = (2, 512)
        x = tf.ones(input_shape)
        r = tf.ones(noise_shape)

        simple_gan_instance = SimpleGAN()
        out = simple_gan_instance(r, x, is_training=True)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose(input_shape, out[0].shape)
            self.assertAllClose((2, 1), out[1].shape)
            self.assertAllClose((2, 1), out[2].shape)

    def test_2d_reg_shape(self):
        input_shape = (2, 64, 64, 1)
        noise_shape = (2, 512)
        x = tf.ones(input_shape)
        r = tf.ones(noise_shape)

        simple_gan_instance = SimpleGAN()
        out = simple_gan_instance(r, x, is_training=True)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose(input_shape, out[0].shape)
            self.assertAllClose((2, 1), out[1].shape)
            self.assertAllClose((2, 1), out[2].shape)



if __name__ == "__main__":
    tf.test.main()
