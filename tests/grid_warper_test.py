from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.layer.grid_warper import _create_affine_features


class GridWarperTest(tf.test.TestCase):

    def test_regular_grids(self):
        out = _create_affine_features([3, 3], [2, 2, 2])
        out = _create_affine_features([3, 3], [2])
        expected = [
            np.array([0., 0., 0., 1., 1., 1., 2., 2., 2.], dtype=np.float32),
            np.array([0., 1., 2., 0., 1., 2., 0., 1., 2.], dtype=np.float32),
            np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=np.float32)]
        self.assertAllClose(expected, out)


if __name__ == "__main__":
    tf.test.main()
