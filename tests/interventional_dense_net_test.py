from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.network.interventional_dense_net import INetDense
from tests.niftynet_testcase import NiftyNetTestCase

class INetDenseTest(NiftyNetTestCase):
    def test_3d_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        densenet_instance = INetDense()
        out = densenet_instance(x, x, is_training=True)
        print(densenet_instance)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 3), out.shape)

    def test_multi_scale_3d_shape(self):
        input_shape = (2, 16, 16, 16, 1)
        x = tf.ones(input_shape)

        densenet_instance = INetDense(multi_scale_fusion=True)
        out = densenet_instance(x, x, is_training=True)
        print(densenet_instance)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 16, 16, 16, 3), out.shape)

    def test_multi_scale_3d_1_shape(self):
        input_shape = (2, 48, 48, 48, 1)
        x = tf.ones(input_shape)

        densenet_instance = INetDense(multi_scale_fusion=True)
        out = densenet_instance(x, x, is_training=True)
        print(densenet_instance)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 48, 48, 48, 3), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 16, 16, 1)
        x = tf.ones(input_shape)

        densenet_instance = INetDense()
        out = densenet_instance(x, x, is_training=True)
        print(densenet_instance)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 16, 16, 2), out.shape)


if __name__ == "__main__":
    tf.test.main()
