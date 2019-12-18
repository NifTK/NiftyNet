from __future__ import absolute_import, print_function

import unittest

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.se_resnet import SE_ResNet
from tests.niftynet_testcase import NiftyNetTestCase

class SeResNet3DTest(NiftyNetTestCase):
    def test_3d_shape(self):
        input_shape = (2, 8, 16, 32, 1)
        x = tf.ones(input_shape)

        resnet_instance = SE_ResNet(num_classes=160)
        out = resnet_instance(x, is_training=True)
        print(resnet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 160), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 8, 16, 1)
        x = tf.ones(input_shape)

        resnet_instance = SE_ResNet(num_classes=160)
        out = resnet_instance(x, is_training=True)
        print(resnet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 160), out.shape)

    def test_3d_reg_shape(self):
        input_shape = (2, 8, 16, 24, 1)
        x = tf.ones(input_shape)

        resnet_instance = SE_ResNet(num_classes=160,
                               w_regularizer=regularizers.l2_regularizer(0.4))
        out = resnet_instance(x, is_training=True)
        print(resnet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 160), out.shape)

    def test_2d_reg_shape(self):
        input_shape = (2, 8, 16, 1)
        x = tf.ones(input_shape)

        resnet_instance = SE_ResNet(num_classes=160,
                               w_regularizer=regularizers.l2_regularizer(0.4))
        out = resnet_instance(x, is_training=True)
        print(resnet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 160), out.shape)


if __name__ == "__main__":
    tf.test.main()
