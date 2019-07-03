from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.network.toynet import ToyNet
from tests.niftynet_testcase import NiftyNetTestCase

class ToyNetTest(NiftyNetTestCase):
    def test_3d_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        toynet_instance = ToyNet(num_classes=160)
        out = toynet_instance(x, is_training=True)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 160), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 32, 32, 1)
        x = tf.ones(input_shape)

        toynet_instance = ToyNet(num_classes=160)
        out = toynet_instance(x, is_training=True)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 160), out.shape)


if __name__ == "__main__":
    tf.test.main()
