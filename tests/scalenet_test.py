from __future__ import absolute_import, print_function

import unittest

import os
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.scalenet import ScaleNet
from tests.niftynet_testcase import NiftyNetTestCase

@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class ScaleNetTest(NiftyNetTestCase):
    def test_3d_shape(self):
        input_shape = (2, 32, 32, 32, 4)
        x = tf.ones(input_shape)

        scalenet_layer = ScaleNet(num_classes=5)
        out = scalenet_layer(x, is_training=True)
        print(scalenet_layer.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 5), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 32, 32, 4)
        x = tf.ones(input_shape)

        scalenet_layer = ScaleNet(num_classes=5)
        out = scalenet_layer(x, is_training=True)
        print(scalenet_layer.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 5), out.shape)

    def test_3d_reg_shape(self):
        input_shape = (2, 32, 32, 32, 4)
        x = tf.ones(input_shape)

        scalenet_layer = ScaleNet(num_classes=5,
                                  w_regularizer=regularizers.l2_regularizer(
                                      0.3))
        out = scalenet_layer(x, is_training=True)
        print(scalenet_layer.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 5), out.shape)

    def test_2d_reg_shape(self):
        input_shape = (2, 32, 32, 4)
        x = tf.ones(input_shape)

        scalenet_layer = ScaleNet(num_classes=5,
                                  w_regularizer=regularizers.l2_regularizer(
                                      0.3))
        out = scalenet_layer(x, is_training=True)
        print(scalenet_layer.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 5), out.shape)


if __name__ == "__main__":
    tf.test.main()
