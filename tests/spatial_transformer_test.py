from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.resampler import ResamplerLayer
from niftynet.layer.spatial_transformer import ResampledFieldGridWarperLayer
from tests.niftynet_testcase import NiftyNetTestCase

class ResamplerTest(NiftyNetTestCase):
    def get_3d_input1(self):
        return tf.expand_dims(tf.constant(
            [[[[1, 2, -1], [3, 4, -2]], [[5, 6, -3], [7, 8, -4]]],
             [[[9, 10, -5], [11, 12, -6]], [[13, 14, -7], [15, 16, -8]]]],
            dtype=tf.float32), 4)

    def get_3d_input2(self):
        return tf.concat([self.get_3d_input1(), 100 + self.get_3d_input1()], 4)

    def _test_correctness(self, input, grid, interpolation, boundary,
                          expected_value):
        resampler = ResamplerLayer(interpolation=interpolation,
                                   boundary=boundary)
        out = resampler(input, grid)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_value = sess.run(out)
            self.assertAllClose(expected_value, out_value)

    def test_interpolation_gridwarper_correctness(self):
        gridL = ResampledFieldGridWarperLayer([2, 2, 3], [2, 2, 2], [2, 2, 2])
        grid = gridL(
            tf.constant([[[[[1, 1, 0], [1, 1, 1]], [[1, 0, 0], [1, 0, 1]]],
                          [[[0, 1, 0], [0, 1, 1]], [[0, 0, 0], [0, 0, 1]]]],
                         [[[[0, 0, 0.5], [0, 0, 1]], [[0, 1, 0.5], [0, 1, 1]]],
                          [[[1, 0, 0.5], [1, 0, 1]],
                           [[1, 1, 0.5], [1, 1, 1]]]]],
                        dtype=tf.float32))
        expected_value = [[[[[7], [8]], [[5], [6]]], [[[3], [4]], [[1], [2]]]],
                          [[[[9.5], [10]], [[11.5], [12]]],
                           [[[13.5], [14]], [[15.5], [16]]]]]
        self._test_correctness(input=self.get_3d_input1(),
                               grid=grid,
                               interpolation='LINEAR',
                               boundary='REPLICATE',
                               expected_value=expected_value)

    def test_interpolation_gridwarper_correctness2(self):
        field = tf.constant(
            [[[[[0, 0, 0.5], [0, 0, 1]], [[0, 1, 0.5], [0, 1, 1]]],
              [[[1, 0, 0.5], [1, 0, 1]], [[1, 1, 0.5], [1, 1, 1]]]]],
            dtype=tf.float32)
        grid = ResampledFieldGridWarperLayer([2, 2, 3], [3, 3, 3], [2, 2, 2])(
            field)
        expected_value = [[[[[9.5], [9.75], [10]], [[10.5], [10.75], [11]],
                            [[11.5], [11.75], [12]]],
                           [[[11.5], [11.75], [12]], [[12.5], [12.75], [13]],
                            [[13.5], [13.75], [14]]],
                           [[[13.5], [13.75], [14]], [[14.5], [14.75], [15]],
                            [[15.5], [15.75], [16]]]]]
        self._test_correctness(input=self.get_3d_input1()[1:, :, :, :, :],
                               grid=grid,
                               interpolation='LINEAR',
                               boundary='REPLICATE',
                               expected_value=expected_value)


if __name__ == "__main__":
    tf.test.main()
