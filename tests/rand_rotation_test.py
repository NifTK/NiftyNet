from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.rand_rotation import RandomRotationLayer
from tests.niftynet_testcase import NiftyNetTestCase


class RandRotationTest(NiftyNetTestCase):
    def get_4d_input(self):
        input_4d = {'testdata': np.ones((16, 16, 16, 8))}
        interp_order = {'testdata': (3,) * 8}
        return input_4d, interp_order

    def get_5d_input(self):
        input_5d = {'testdata': np.ones((32, 32, 32, 8, 1))}
        interp_order = {'testdata': (3,)}
        return input_5d, interp_order

    def test_4d_shape(self):
        x, interp_orders = self.get_4d_input()
        rand_rotation_layer = RandomRotationLayer()
        rand_rotation_layer.init_uniform_angle((-10.0, 10.0))
        rand_rotation_layer.randomise()
        out = rand_rotation_layer(x, interp_orders)

    def test_5d_shape(self):
        x, interp_orders = self.get_5d_input()
        rand_rotation_layer = RandomRotationLayer()
        rand_rotation_layer.init_uniform_angle((-10.0, 10.0))
        rand_rotation_layer.randomise()
        out = rand_rotation_layer(x, interp_orders)


if __name__ == "__main__":
    tf.test.main()
