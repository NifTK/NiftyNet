from __future__ import absolute_import, print_function

import unittest

import os
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.scalenet import ScaleBlock
from tests.niftynet_testcase import NiftyNetTestCase

@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class ScaleBlockTest(NiftyNetTestCase):
    def get_2d_input(self):
        input_shape = (2, 32, 32, 4)
        x = tf.ones(input_shape)
        x = tf.unstack(x, axis=-1)
        for (idx, fea) in enumerate(x):
            x[idx] = tf.expand_dims(fea, axis=-1)
        x = tf.stack(x, axis=-1)
        return x

    def get_3d_input(self):
        input_shape = (2, 32, 32, 32, 4)
        x = tf.ones(input_shape)
        x = tf.unstack(x, axis=-1)
        for (idx, fea) in enumerate(x):
            x[idx] = tf.expand_dims(fea, axis=-1)
        x = tf.stack(x, axis=-1)
        return x

    def test_2d_shape(self):
        x = self.get_2d_input()
        scalenet_layer = ScaleBlock('AVERAGE', n_layers=1)
        out_1 = scalenet_layer(x, is_training=True)
        print(scalenet_layer)

        scalenet_layer = ScaleBlock('MAX', n_layers=2)
        out_2 = scalenet_layer(x, is_training=True)
        print(scalenet_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 32, 32, 1), out_1.shape)
            self.assertAllClose((2, 32, 32, 1), out_2.shape)

    def test_2d_reg_shape(self):
        x = self.get_2d_input()
        scalenet_layer = ScaleBlock(
            'AVERAGE',
            n_layers=1,
            w_regularizer=regularizers.l2_regularizer(0.3))
        out_1 = scalenet_layer(x, is_training=True)
        print(scalenet_layer)

        scalenet_layer = ScaleBlock('MAX', n_layers=2)
        out_2 = scalenet_layer(x, is_training=True)
        print(scalenet_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 32, 32, 1), out_1.shape)
            self.assertAllClose((2, 32, 32, 1), out_2.shape)

    def test_3d_shape(self):
        x = self.get_3d_input()
        scalenet_layer = ScaleBlock('AVERAGE', n_layers=1)
        out_1 = scalenet_layer(x, is_training=True)
        print(scalenet_layer)

        scalenet_layer = ScaleBlock('MAX', n_layers=2)
        out_2 = scalenet_layer(x, is_training=True)
        print(scalenet_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 32, 32, 32, 1), out_1.shape)
            self.assertAllClose((2, 32, 32, 32, 1), out_2.shape)

    def test_3d_reg_shape(self):
        x = self.get_3d_input()
        scalenet_layer = ScaleBlock(
            'AVERAGE',
            n_layers=1,
            w_regularizer=regularizers.l2_regularizer(0.3))
        out_1 = scalenet_layer(x, is_training=True)
        print(scalenet_layer)

        scalenet_layer = ScaleBlock('MAX', n_layers=2)
        out_2 = scalenet_layer(x, is_training=True)
        print(scalenet_layer)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            out_2 = sess.run(out_2)
            self.assertAllClose((2, 32, 32, 32, 1), out_1.shape)
            self.assertAllClose((2, 32, 32, 32, 1), out_2.shape)


if __name__ == "__main__":
    tf.test.main()
