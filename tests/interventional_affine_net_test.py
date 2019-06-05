from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.network.toynet import ToyNet
from niftynet.network.interventional_affine_net import INetAffine
from tests.niftynet_testcase import NiftyNetTestCase


class INetAffineTest(NiftyNetTestCase):
    def test_3d_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        affinenet_instance = INetAffine()
        out = affinenet_instance(x, x, is_training=True)
        print(affinenet_instance)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 3), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 32, 32, 1)
        x = tf.ones(input_shape)

        affinenet_instance = INetAffine()
        out = affinenet_instance(x, x, is_training=True)
        print(affinenet_instance)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 2), out.shape)


if __name__ == "__main__":
    tf.test.main()
