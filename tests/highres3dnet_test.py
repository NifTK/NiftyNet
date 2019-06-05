from __future__ import absolute_import, print_function

import os
import unittest

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.highres3dnet import HighRes3DNet
from niftynet.network.highres3dnet_large import HighRes3DNetLarge
from niftynet.network.highres3dnet_small import HighRes3DNetSmall
from tests.niftynet_testcase import NiftyNetTestCase


@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true",
                 'Skipping slow tests')
class HighRes3DNetTest(NiftyNetTestCase):
    def shape_test(self, input_shape, expected_shape):
        x = tf.ones(input_shape)

        highres_layer = HighRes3DNet(num_classes=5)
        highres_layer_small = HighRes3DNetSmall(num_classes=5)
        highres_layer_large = HighRes3DNetLarge(num_classes=5)

        out = highres_layer(x, is_training=True)
        out_small = highres_layer_small(x, is_training=True)
        out_large = highres_layer_large(x, is_training=True)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out, out_large, out_small = sess.run([out, out_large, out_small])
            self.assertAllClose(expected_shape, out.shape)
            self.assertAllClose(expected_shape, out_large.shape)
            self.assertAllClose(expected_shape, out_small.shape)

    def shape_test_reg(self, input_shape, expected_shape):
        x = tf.ones(input_shape)
        layer_param = {
            'num_classes': 5,
            'w_regularizer': regularizers.l2_regularizer(0.5),
            'b_regularizer': regularizers.l2_regularizer(0.5)}

        highres_layer = HighRes3DNet(**layer_param)
        highres_layer_small = HighRes3DNetSmall(**layer_param)
        highres_layer_large = HighRes3DNetLarge(**layer_param)

        out = highres_layer(x, is_training=True)
        out_small = highres_layer_small(x, is_training=True)
        out_large = highres_layer_large(x, is_training=True)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out, out_large, out_small = sess.run([out, out_large, out_small])
            self.assertAllClose(expected_shape, out.shape)
            self.assertAllClose(expected_shape, out_large.shape)
            self.assertAllClose(expected_shape, out_small.shape)

    def test_2d(self):
        self.shape_test(input_shape=(2, 32, 32, 1),
                        expected_shape=(2, 32, 32, 5))
        self.shape_test_reg(input_shape=(2, 32, 32, 1),
                            expected_shape=(2, 32, 32, 5))

    def test_3d(self):
        self.shape_test(input_shape=(2, 32, 32, 32, 1),
                        expected_shape=(2, 32, 32, 32, 5))
        self.shape_test_reg(input_shape=(2, 32, 32, 32, 1),
                            expected_shape=(2, 32, 32, 32, 5))


if __name__ == "__main__":
    tf.test.main()
