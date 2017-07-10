from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.utilities.subject import ColumnData


class RandRotationTest(tf.test.TestCase):
    def get_4d_input(self):
        input_4d = ColumnData('testdata',
                              np.ones((16, 16, 16, 8)),
                              spatial_rank=3,
                              interp_order=3)
        return input_4d

    def get_5d_input(self):
        input_5d = ColumnData('testdata',
                              np.ones((32, 32, 32, 8, 1)),
                              3, 3)
        return input_5d

    def test_4d_shape(self):
        x = self.get_4d_input()
        rand_layer = RandomSpatialScalingLayer(min_percentage=-10.0,
                                               max_percentage=10.0)
        rand_layer.randomise(x.spatial_rank)
        out = rand_layer(x)

    def test_5d_shape(self):
        x = self.get_5d_input()
        rand_layer = RandomSpatialScalingLayer(min_percentage=-10.0,
                                               max_percentage=10.0)
        rand_layer.randomise(x.spatial_rank)
        out = rand_layer(x)

if __name__ == "__main__":
    tf.test.main()
