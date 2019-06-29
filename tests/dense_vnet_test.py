from __future__ import absolute_import, print_function

import unittest

import os
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.dense_vnet import DenseVNet
from tests.niftynet_testcase import NiftyNetTestCase

@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class DenseVNetTest(NiftyNetTestCase):
    def test_3d_shape(self):
        input_shape = (2, 72, 72, 72, 3)
        x = tf.ones(input_shape)

        dense_vnet_instance = DenseVNet(
            num_classes=2)
        out = dense_vnet_instance(x, is_training=True)
        # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 72, 72, 72, 2), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 72, 72, 3)
        x = tf.ones(input_shape)

        dense_vnet_instance = DenseVNet(
            num_classes=2)
        out = dense_vnet_instance(x, is_training=True)
        # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 72, 72, 2), out.shape)

if __name__ == "__main__":
    tf.test.main()
