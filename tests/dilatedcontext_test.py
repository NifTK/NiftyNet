from __future__ import absolute_import, print_function
import tensorflow as tf

from niftynet.layer.dilatedcontext import DilatedTensor
from tests.niftynet_testcase import NiftyNetTestCase


class BNTest(NiftyNetTestCase):
    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def test_2d_dilating_shape(self):
        x = self.get_2d_input()
        with DilatedTensor(x, 4) as dilated:
            intermediate = dilated.tensor
        x = dilated.tensor

        with self.cached_session() as sess:
            out = sess.run(x)
            out_dilated = sess.run(intermediate)
            self.assertAllClose((2, 16, 16, 8), out.shape)
            self.assertAllClose((32, 4, 4, 8), out_dilated.shape)

    def test_3d_dilating_shape(self):
        x = self.get_3d_input()
        with DilatedTensor(x, 4) as dilated:
            intermediate = dilated.tensor
        x = dilated.tensor

        with self.cached_session() as sess:
            out = sess.run(x)
            out_dilated = sess.run(intermediate)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)
            self.assertAllClose((128, 4, 4, 4, 8), out_dilated.shape)


if __name__ == "__main__":
    tf.test.main()
