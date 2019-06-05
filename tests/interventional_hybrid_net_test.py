from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.network.toynet import ToyNet
from niftynet.network.interventional_hybrid_net import INetHybridPreWarp, INetHybridTwoStream
from tests.niftynet_testcase import NiftyNetTestCase


class INetHybridPreWarpTest(NiftyNetTestCase):

    def test_3d_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        hybridnet_instance = INetHybridPreWarp(1e-6)
        out = hybridnet_instance(x, x, is_training=True)
        print(hybridnet_instance)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 3), out[0].shape)
            self.assertAllClose((2, 32, 32, 32, 3), out[1].shape)

    def test_2d_shape(self):
        input_shape = (2, 32, 32, 1)
        x = tf.ones(input_shape)

        hybridnet_instance = INetHybridPreWarp(1e-6)
        out = hybridnet_instance(x, x, is_training=True)
        print(hybridnet_instance)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 2), out[0].shape)
            self.assertAllClose((2, 32, 32, 2), out[1].shape)

class INetHybridTwoStreamTest(NiftyNetTestCase):


    def test_3d_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        hybridnet_instance = INetHybridTwoStream(1e-6)
        out = hybridnet_instance(x, x, is_training=True)
        print(hybridnet_instance)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 3), out[0].shape)
            self.assertAllClose((2, 32, 32, 32, 3), out[1].shape)
            self.assertAllClose((2, 32, 32, 32, 3), out[2].shape)

    def test_2d_shape(self):
        input_shape = (2, 32, 32, 1)
        x = tf.ones(input_shape)

        hybridnet_instance = INetHybridTwoStream(1e-6)
        out = hybridnet_instance(x, x, is_training=True)
        print(hybridnet_instance)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 2), out[0].shape)
            self.assertAllClose((2, 32, 32, 2), out[1].shape)
            self.assertAllClose((2, 32, 32, 2), out[2].shape)


if __name__ == "__main__":
    tf.test.main()
