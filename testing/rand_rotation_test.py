import tensorflow as tf
import numpy as np

from layer.rand_rotation import RandomRotationLayer


class RandRotationTest(tf.test.TestCase):
    def get_4d_input(self):
        input_4d = np.ones((16, 16, 16, 8))
        return input_4d

    def get_5d_input(self):
        input_5d = np.ones((32, 32, 32, 8, 1))
        return input_5d

    def test_4d_shape(self):
        x = self.get_4d_input()
        rand_rotation_layer = RandomRotationLayer(
            min_angle=-10.0, max_angle=10.0)
        rand_rotation_layer.randomise_transformation()
        out = rand_rotation_layer(x, spatial_rank=3, interp_order=3)
        self.assertAllClose(x.shape, out.shape)

    def test_5d_shape(self):
        x = self.get_5d_input()
        rand_rotation_layer = RandomRotationLayer(
            min_angle=-10.0, max_angle=10.0)
        rand_rotation_layer.randomise_transformation()
        out = rand_rotation_layer(x, spatial_rank=3, interp_order=0)
        self.assertAllClose(x.shape, out.shape)

if __name__ == "__main__":
    tf.test.main()
