from __future__ import absolute_import, print_function, division

import tensorflow as tf

from niftynet.layer.resampler import ResamplerLayer
from niftynet.layer.grid_warper import AffineGridWarperLayer

class ResamplerGridWarperTest(tf.test.TestCase):
    def get_3d_input1(self):
        test_case = tf.constant(
            [[[[1, 2, -1], [3, 4, -2]], [[5, 6, -3], [7, 8, -4]]],
             [[[9, 10, -5], [11, 12, -6]], [[13, 14, -7], [15, 16, -8]]]],
            dtype=tf.float32)
        return tf.expand_dims(test_case, 4)

    def _test_correctness(
        self, input, grid, interpolation, boundary, expected_value):
        resampler = ResamplerLayer(interpolation=interpolation,
                                   boundary=boundary)
        out = resampler(input, grid)
        with self.test_session() as sess:
            out_value = sess.run(out)
            self.assertAllClose(expected_value, out_value)

    def test_combined(self):
        expected = [[[[[1], [2]], [[3], [4]]],
                     [[[5], [6]], [[7], [8]]]],
                    [[[[9.5], [2.5]], [[11.5], [3]]],
                     [[[13.5], [3.5]], [[15.5], [4]]]]]
        affine_grid = AffineGridWarperLayer(source_shape=(2, 2, 3),
                                            output_shape=(2, 2, 2))
        test_grid = affine_grid(
            tf.constant([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                         [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, .5]],
                         dtype=tf.float32))
        self._test_correctness(input=self.get_3d_input1(),
                               grid=test_grid,
                               interpolation='idw',
                               boundary='replicate',
                               expected_value=expected)


if __name__ == "__main__":
    tf.test.main()
