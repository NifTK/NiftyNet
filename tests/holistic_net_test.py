from __future__ import absolute_import, print_function

import unittest

import os
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.holistic_net import HolisticNet
from tests.niftynet_testcase import NiftyNetTestCase


class HolisticNetTest(NiftyNetTestCase):
    def test_3d_reg_shape(self):
        input_shape = (2, 20, 20, 20, 1)
        x = tf.ones(input_shape)

        holistic_net_instance = HolisticNet(
            num_classes=3,
            w_regularizer=regularizers.l2_regularizer(0.5),
            b_regularizer=regularizers.l2_regularizer(0.5))
        out = holistic_net_instance(x, is_training=False)
        # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 20, 20, 20, 3), out.shape)

    def test_2d_reg_shape(self):
        input_shape = (2, 20, 20, 1)
        x = tf.ones(input_shape)

        holistic_net_instance = HolisticNet(
            num_classes=3,
            w_regularizer=regularizers.l2_regularizer(0.5),
            b_regularizer=regularizers.l2_regularizer(0.5))
        out = holistic_net_instance(x, is_training=False)
        # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 20, 20, 3), out.shape)



if __name__ == "__main__":
    tf.test.main()
