from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import regularizers

from layer.convolution import ConvLayer
from layer.spatial_transformer import ResamplerLayer, AffineGridWarperLayer, ResampledFieldGridWarperLayer


class ResamplerTest(tf.test.TestCase):
    def get_3d_input1(self):
        return tf.expand_dims(tf.constant(
            [[[[1, 2, -1], [3, 4, -2]], [[5, 6, -3], [7, 8, -4]]],
             [[[9, 10, -5], [11, 12, -6]], [[13, 14, -7], [15, 16, -8]]]],
            dtype=tf.float32), 4)

    def get_3d_input2(self):
        return tf.concat([self.get_3d_input1(), 100 + self.get_3d_input1()], 4)

    def get_identity_warp(self):
        params = tf.tile(tf.constant([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]],
                                     dtype=tf.float32), [sz[0], 1])
        grid = AffineGridWarperLayer(source_shape=sz[1:4], output_shape=sz[1:4])
        return grid, params

    def _test_correctness(self, input, grid, interpolation, boundary,
                          expected_value):
        resampler = ResamplerLayer(interpolation=interpolation,
                                   boundary=boundary)
        out = resampler(input, grid)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_value = sess.run(out)
            self.assertAllClose(expected_value, out_value)

    def test_resampler_3d_multivariate_replicate_linear_correctness(self):
        self._test_correctness(input=self.get_3d_input2(),
                               grid=tf.constant(
                                   [[[.25, .25, .25], [.25, .75, .25]],
                                    [[.75, .25, .25], [.25, .25, .75]]],
                                   dtype=tf.float32),
                               interpolation='LINEAR',
                               boundary='REPLICATE',
                               expected_value=[[[2.75, 102.75], [3.75, 103.75]],
                                               [[12.75, 112.75],
                                                [11.25, 111.25]]])

    def test_resampler_3d_replicate_nearest_correctness(self):
        self._test_correctness(input=self.get_3d_input2(),
                               grid=tf.constant(
                                   [[[.25, .25, .25], [.25, .75, .25]],
                                    [[.75, .25, .25], [.25, .25, .75]]],
                                   dtype=tf.float32),
                               interpolation='NEAREST',
                               boundary='REPLICATE',
                               expected_value=[[[1, 101], [3, 103]],
                                               [[13, 113], [10, 110]]])

    def test_resampler_3d_replicate_linear_correctness(self):
        self._test_correctness(input=self.get_3d_input1(),
                               grid=tf.constant(
                                   [[[.25, .25, .25], [.25, .75, .25]],
                                    [[.75, .25, .25], [.25, .25, .75]]],
                                   dtype=tf.float32),
                               interpolation='LINEAR',
                               boundary='REPLICATE',
                               expected_value=[[[2.75], [3.75]],
                                               [[12.75], [11.25]]])

    def test_resampler_3d_replicate_cubic_correctness(self):
        self._test_correctness(input=self.get_3d_input1(),
                               grid=tf.constant(
                                   [[[.25,.25,.25],[.25,.75,.25]],
                                   [[.75,.25,.25],[.25,.25,.75]]],
                                   dtype=tf.float32),
                               interpolation='BSPLINE',
                               boundary='REPLICATE',
                               expected_value=[[[3.20869954],[3.93501790]],
                                               [[12.63008626],[10.33280436]]])
    def test_resampler_3d_replicate_nearest_correctness(self):
        self._test_correctness(input=self.get_3d_input1(),
                               grid=tf.constant(
                                   [[[.25, .25, .25], [.25, .75, .25]],
                                    [[.75, .25, .25], [.25, .25, .75]]],
                                   dtype=tf.float32),
                               interpolation='NEAREST',
                               boundary='REPLICATE',
                               expected_value=[[[1], [3]], [[13], [10]]])

    def test_resampler_3d_circular_linear_correctness(self):
        self._test_correctness(input=self.get_3d_input1(),
                               grid=tf.constant([[[.25, .25 + 2, .25 + 3],
                                                  [.25 - 2, .75 - 2, .25 - 3]],
                                                 [[.75 + 2, .25 - 2, .25 - 3],
                                                  [.25 + 2, .25 - 2, .75 + 3]]],
                                                dtype=tf.float32),
                               interpolation='LINEAR',
                               boundary='CIRCULAR',
                               expected_value=[[[2.75], [3.75]],
                                               [[12.75], [11.25]]])

    def test_resampler_3d_circular_nearest_correctness(self):
        self._test_correctness(input=self.get_3d_input1(),
                               grid=tf.constant([[[.25, .25 + 2, .25 + 3],
                                                  [.25 - 2, .75 - 2, .25 - 3]],
                                                 [[.75 + 4, .25 - 6, .25 - 6],
                                                  [.25 + 2, .25 - 4, .75 + 9]]],
                                                dtype=tf.float32),
                               interpolation='NEAREST',
                               boundary='CIRCULAR',
                               expected_value=[[[1], [3]], [[13], [10]]])

    def test_resampler_3d_symmetric_linear_correctness(self):
        self._test_correctness(input=self.get_3d_input1(),
                               grid=tf.constant([[[-.25, -.25, -.25],
                                                  [.25 + 2, .75 + 2, .25 + 4]],
                                                 [[.75, .25, -.25 + 4],
                                                  [.25, .25, .75]]],
                                                dtype=tf.float32),
                               interpolation='LINEAR',
                               boundary='SYMMETRIC',
                               expected_value=[[[2.75], [3.75]],
                                               [[12.75], [11.25]]])

    def test_resampler_3d_symmetric_nearest_correctness(self):
        self._test_correctness(input=self.get_3d_input1(),
                               grid=tf.constant([[[-.25, -.25, -.25],
                                                  [.25 + 2, .75 + 2, .25 + 4]],
                                                 [[.75, .25, -.25 + 4],
                                                  [.25, .25, .75]]],
                                                dtype=tf.float32),
                               interpolation='NEAREST',
                               boundary='SYMMETRIC',
                               expected_value=[[[1], [3]], [[13], [10]]])

    def test_resampler_gridwarper_combination_correctness(self):
        self._test_correctness(input=self.get_3d_input1(),
                               grid=AffineGridWarperLayer([2, 2, 3], [2, 2, 2])(
                                   tf.constant(
                                       [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, .5]],
                                       dtype=tf.float32)),
                               interpolation='LINEAR',
                               boundary='REPLICATE',
                               expected_value=[[[[[1], [2]], [[3], [4]]],
                                                [[[5], [6]], [[7], [8]]]],
                                               [[[[9.5], [2.5]], [[11.5], [3]]],
                                                [[[13.5], [3.5]],
                                                 [[15.5], [4]]]]])

    def test_interpolation_gridwarper_correctness(self):
        gridL = ResampledFieldGridWarperLayer([2,2,3],[2,2,2],[2,2,2])
        grid=gridL(tf.constant([[[[[1,1,0],[1,1,1]],[[1,0,0],[1,0,1]]],
                                 [[[0,1,0],[0,1,1]],[[0,0,0],[0,0,1]]]],
                                [[[[0,0,0.5],[0,0,1]],[[0,1,0.5],[0,1,1]]],
                                 [[[1,0,0.5],[1,0,1]],[[1,1,0.5],[1,1,1]]]]],
                               dtype=tf.float32))
        expected_value = [[[[[7],[8]],[[5],[6]]],[[[3],[4]],[[1],[2]]]],
                          [[[[9.5],[10]],[[11.5],[12]]],[[[13.5],[14]],[[15.5],[16]]]]]
        self._test_correctness(input=self.get_3d_input1(),
                               grid=grid,
                               interpolation='LINEAR',
                               boundary='REPLICATE',
                               expected_value=expected_value)
    def test_interpolation_gridwarper_correctness2(self):
        field=tf.constant([[[[[0,0,0.5],[0,0,1]],[[0,1,0.5],[0,1,1]]],
                            [[[1,0,0.5],[1,0,1]],[[1,1,0.5],[1,1,1]]]]],dtype=tf.float32)
        grid = ResampledFieldGridWarperLayer([2,2,3],[3,3,3],[2,2,2])(field)
        expected_value = [[[[[9.5],[9.75],[10]],[[10.5],[10.75],[11]],[[11.5],[11.75],[12]]],
                           [[[11.5],[11.75],[12]],[[12.5],[12.75],[13]],[[13.5],[13.75],[14]]],
                           [[[13.5],[13.75],[14]],[[14.5],[14.75],[15]],[[15.5],[15.75],[16]]]]]
        self._test_correctness(input=self.get_3d_input1()[1:,:,:,:,:],
                               grid=grid,
                               interpolation='LINEAR',
                               boundary='REPLICATE',
                               expected_value=expected_value)

if __name__ == "__main__":
    tf.test.main()
