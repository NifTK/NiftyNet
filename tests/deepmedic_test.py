from __future__ import absolute_import, print_function

import unittest

import os
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.deepmedic import DeepMedic
from tests.niftynet_testcase import NiftyNetTestCase


@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class DeepMedicTest(NiftyNetTestCase):
    def test_3d_reg_shape(self):
        input_shape = (2, 57, 57, 57, 1)
        x = tf.ones(input_shape)

        deepmedic_instance = DeepMedic(
            num_classes=160,
            w_regularizer=regularizers.l2_regularizer(0.5),
            b_regularizer=regularizers.l2_regularizer(0.5))
        out = deepmedic_instance(x, is_training=True)
        # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 9, 9, 9, 160), out.shape)

    def test_2d_reg_shape(self):
        input_shape = (2, 57, 57, 1)
        x = tf.ones(input_shape)

        deepmedic_instance = DeepMedic(
            num_classes=160,
            w_regularizer=regularizers.l2_regularizer(0.5),
            b_regularizer=regularizers.l2_regularizer(0.5))
        out = deepmedic_instance(x, is_training=True)
        # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 9, 9, 160), out.shape)

    def test_3d_shape(self):
        input_shape = (2, 57, 57, 57, 1)
        x = tf.ones(input_shape)

        deepmedic_instance = DeepMedic(num_classes=160)
        out = deepmedic_instance(x, is_training=True)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 9, 9, 9, 160), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 57, 57, 1)
        x = tf.ones(input_shape)

        deepmedic_instance = DeepMedic(num_classes=160)
        out = deepmedic_instance(x, is_training=True)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 9, 9, 160), out.shape)


if __name__ == "__main__":
    tf.test.main()
