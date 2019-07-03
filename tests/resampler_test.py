from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.layer.resampler import ResamplerLayer
from tests.niftynet_testcase import NiftyNetTestCase

class ResamplerTest(NiftyNetTestCase):
    def get_2d_input(self, as_tensor=True):
        test_array = np.array(
            [[[[1, 2, -1], [3, 4, -2]], [[5, 6, -3], [7, 8, -4]]],
             [[[9, 10, -5], [11, 12, -6]], [[13, 14, -7], [15, 16, -8]]]])
        if as_tensor:
            test_array = tf.constant(test_array, dtype=tf.float32)
            return test_array
        return test_array.astype(np.float32)

    def get_3d_input1(self, as_tensor=True):
        test_array = np.array(
            [[[[1, 2, -1], [3, 4, -2]], [[5, 6, -3], [7, 8, -4]]],
             [[[9, 10, -5], [11, 12, -6]], [[13, 14, -7], [15, 16, -8]]]])
        if as_tensor:
            test_array = tf.constant(test_array, dtype=tf.float32)
            return tf.expand_dims(test_array, 4)
        return np.expand_dims(test_array, 4).astype(np.float32)

    def get_3d_input2(self, as_tensor=True):
        one_channel = self.get_3d_input1(as_tensor=as_tensor)
        if as_tensor:
            return tf.concat([one_channel, 100 + one_channel], 4)
        return np.concatenate([one_channel, 100 + one_channel], 4)

    def _test_correctness(
            self, input, grid, interpolation, boundary, expected_value):
        resampler = ResamplerLayer(interpolation=interpolation,
                                   boundary=boundary)
        out = resampler(input, grid)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            # print(expected_value)
            # print(out_value)
            self.assertAllClose(expected_value, out_value)

    def test_resampler_3d_multivariate_zero_weight_idw_correctness(self):
        test_grid = tf.constant(
            [[[0, 1, 2], [.25, .75, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[-2, 98], [3.87344956, 103.873459]],
                    [[12.70884895, 112.70884705], [11.45574856, 111.45578003]]]
        self._test_correctness(input=self.get_3d_input2(),
                               grid=test_grid,
                               interpolation='IDW',
                               boundary='REPLICATE',
                               expected_value=expected)

    def test_resampler_3d_multivariate_replicate_idw_correctness(self):
        test_grid = tf.constant(
            [[[.25, .25, .25], [.25, .75, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[3.03804874, 103.03804016], [3.87344956, 103.87354279]],
                    [[12.70884895, 112.70884705], [11.45574856, 111.45578003]]]
        self._test_correctness(input=self.get_3d_input2(),
                               grid=test_grid,
                               interpolation='IDW',
                               boundary='REPLICATE',
                               expected_value=expected)

    def test_resampler_2d_replicate_linear_correctness(self):
        test_grid = tf.constant(
            [[[.25, .25], [.25, .78]],
             [[.62, .25], [.25, .28]]],
            dtype=tf.float32)
        expected = [[[2.5, 3.5, -1.75],
                     [3.56, 4.56, -2.28]],
                    [[11.98, 12.98, -6.49],
                     [10.56, 11.56, -5.78]]]
        self._test_correctness(input=self.get_2d_input(),
                               grid=test_grid,
                               interpolation='LINEAR',
                               boundary='ZERO',
                               expected_value=expected)

    def test_resampler_3d_multivariate_replicate_linear_correctness(self):
        test_grid = tf.constant(
            [[[.25, .25, .25], [.25, .75, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[2.75, 102.75], [3.75, 103.75]],
                    [[12.75, 112.75], [11.25, 111.25]]]
        self._test_correctness(input=self.get_3d_input2(),
                               grid=test_grid,
                               interpolation='LINEAR',
                               boundary='REPLICATE',
                               expected_value=expected)

    def test_resampler_3d_replicate_nearest_correctness(self):
        test_grid = tf.constant(
            [[[.25, .25, .25], [.25, .75, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[1, 101], [3, 103]],
                    [[13, 113], [10, 110]]]
        self._test_correctness(input=self.get_3d_input2(),
                               grid=test_grid,
                               interpolation='NEAREST',
                               boundary='REPLICATE',
                               expected_value=expected)

    def test_resampler_3d_zero_nearest_correctness(self):
        test_grid = tf.constant(
            [[[-5.2, .25, .25], [.25, .95, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[0, 0], [3, 103]],
                    [[13, 113], [10, 110]]]
        self._test_correctness(input=self.get_3d_input2(),
                               grid=test_grid,
                               interpolation='NEAREST',
                               boundary='ZERO',
                               expected_value=expected)

    def test_resampler_3d_replicate_linear_correctness(self):
        test_grid = tf.constant(
            [[[.25, .25, .25], [.25, .75, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[2.75], [3.75]],
                    [[12.75], [11.25]]]
        self._test_correctness(input=self.get_3d_input1(),
                               grid=test_grid,
                               interpolation='LINEAR',
                               boundary='REPLICATE',
                               expected_value=expected)

    def test_resampler_3d_replicate_cubic_correctness(self):
        test_grid = tf.constant(
            [[[.25, .25, .25], [.25, .75, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[3.20869954], [3.93501790]],
                    [[12.63008626], [10.33280436]]]
        self._test_correctness(input=self.get_3d_input1(),
                               grid=test_grid,
                               interpolation='BSPLINE',
                               boundary='REPLICATE',
                               expected_value=expected)

    def test_resampler_3d_circular_nearest_correctness(self):
        test_grid = tf.constant(
            [[[.25, .25, .25], [.25, .75, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[1], [3]], [[13], [10]]]
        self._test_correctness(input=self.get_3d_input1(),
                               grid=test_grid,
                               interpolation='NEAREST',
                               boundary='CIRCULAR',
                               expected_value=expected)

    def test_resampler_3d_circular_linear_correctness(self):
        test_grid = tf.constant(
            [[[.25, .25 + 2, .25 + 3],
              [.25 - 2, .75 - 2, .25 - 3]],
             [[.75 + 2, .25 - 2, .25 - 3],
              [.25 + 2, .25 - 2, .75 + 3]]],
            dtype=tf.float32)
        expected = [[[2.75], [3.75]],
                    [[12.75], [11.25]]]
        self._test_correctness(input=self.get_3d_input1(),
                               grid=test_grid,
                               interpolation='LINEAR',
                               boundary='CIRCULAR',
                               expected_value=expected)

    def test_resampler_3d_symmetric_nearest_correctness(self):
        test_grid = tf.constant(
            [[[-.25, -.25, -.25],
              [.25 + 2, .75 + 2, .25 + 4]],
             [[.75, .25, -.25 + 4],
              [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[1], [3]], [[13], [10]]]
        self._test_correctness(input=self.get_3d_input1(),
                               grid=test_grid,
                               interpolation='NEAREST',
                               boundary='SYMMETRIC',
                               expected_value=expected)

    def test_resampler_3d_symmetric_linear_correctness(self):
        test_grid = tf.constant(
            [[[-.25, -.25, -.25],
              [.25 + 2, .75 + 2, .25 + 4]],
             [[.75, .25, -.25 + 4],
              [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[2.75], [3.75]],
                    [[12.75], [11.25]]]
        self._test_correctness(input=self.get_3d_input1(),
                               grid=test_grid,
                               interpolation='LINEAR',
                               boundary='SYMMETRIC',
                               expected_value=expected)

    def test_resampler_3d_symmetric_cubic_correctness(self):
        test_grid = tf.constant(
            [[[-.25, -.25, -.25],
              [.25 + 2, .75 + 2, .25 + 4]],
             [[.75, .25, -.25 + 4],
              [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[3.683675], [4.140218]],
                    [[12.56551075], [10.69881153]]]
        self._test_correctness(input=self.get_3d_input1(),
                               grid=test_grid,
                               interpolation='BSPLINE',
                               boundary='SYMMETRIC',
                               expected_value=expected)

    def test_resampler_3d_circular_cubic_correctness(self):
        test_grid = tf.constant(
            [[[-.25, -.25, -.25],
              [.25 + 2, .75 + 2, .25 + 4]],
             [[.75, .25, -.25 + 4],
              [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[1.66219068], [2.44295263]],
                    [[11.46712303], [10.65071392]]]
        self._test_correctness(input=self.get_3d_input1(),
                               grid=test_grid,
                               interpolation='BSPLINE',
                               boundary='CIRCULAR',
                               expected_value=expected)

    def _test_partial_shape_correctness(self,
                                        input,
                                        rank,
                                        batch_size,
                                        grid,
                                        interpolation,
                                        boundary,
                                        expected_value=None):

        resampler = ResamplerLayer(interpolation=interpolation,
                                   boundary=boundary)
        input_default = tf.random_uniform(input.shape)
        if batch_size > 0 and rank > 0:
            input_placeholder = tf.placeholder_with_default(
                input_default, shape=[batch_size] + [None] * (rank + 1))
        elif batch_size <= 0 and rank > 0:
            input_placeholder = tf.placeholder_with_default(
                input_default, shape=[None] * (rank + 2))
        elif batch_size <= 0 and rank <= 0:
            input_placeholder = tf.placeholder_with_default(
                input_default, shape=None)

        out = resampler(input_placeholder, grid)
        with self.cached_session() as sess:
            out_value = sess.run(
                out, feed_dict={input_placeholder: input})
            # print(expected_value)
            # print(out_value)
            if expected_value is not None:
                self.assertAllClose(expected_value, out_value)

    def test_2d_linear_partial_shapes(self):
        test_grid = tf.constant(
            [[[.25, .25], [.25, .78]],
             [[.62, .25], [.25, .28]]], dtype=tf.float32)
        expected = [[[2.5, 3.5, -1.75],
                     [3.56, 4.56, -2.28]],
                    [[11.98, 12.98, -6.49],
                     [10.56, 11.56, -5.78]]]
        interp = 'linear'

        from niftynet.layer.resampler import SUPPORTED_BOUNDARY
        for b in list(SUPPORTED_BOUNDARY):
            self._test_partial_shape_correctness(
                input=self.get_2d_input(False),
                rank=2,
                batch_size=2,
                grid=test_grid,
                interpolation=interp,
                boundary=b,
                expected_value=expected)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_2d_input(False),
                    rank=2,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_2d_input(False),
                    rank=-1,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)

    def test_3d_linear_partial_shapes(self):
        test_grid = tf.constant(
            [[[.25, .25, .25], [.25, .75, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[2.75, 102.75], [3.75, 103.75]],
                    [[12.75, 112.75], [11.25, 111.25]]]
        interp = 'linear'

        from niftynet.layer.resampler import SUPPORTED_BOUNDARY
        for b in list(SUPPORTED_BOUNDARY):
            self._test_partial_shape_correctness(
                input=self.get_3d_input2(False),
                rank=3,
                batch_size=2,
                grid=test_grid,
                interpolation=interp,
                boundary=b,
                expected_value=expected)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_3d_input2(False),
                    rank=3,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_3d_input2(False),
                    rank=-1,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)

    def test_2d_idw_partial_shapes(self):
        test_grid = tf.constant(
            [[[.25, .25], [.25, .78]],
             [[.62, .25], [.25, .28]]], dtype=tf.float32)
        expected = [[[2.23529005, 3.23529005, -1.61764503],
                     [3.40578985, 4.40578985, -2.20289493]],
                    [[12.13710022, 13.13710022, -6.56855011],
                     [10.34773731, 11.34773731, -5.67386866]]]
        interp = 'IDW'

        from niftynet.layer.resampler import SUPPORTED_BOUNDARY
        for b in list(SUPPORTED_BOUNDARY):
            self._test_partial_shape_correctness(
                input=self.get_2d_input(False),
                rank=2,
                batch_size=2,
                grid=test_grid,
                interpolation=interp,
                boundary=b,
                expected_value=expected)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_2d_input(False),
                    rank=2,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_2d_input(False),
                    rank=-1,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)

    def test_3d_idw_partial_shapes(self):
        test_grid = tf.constant(
            [[[0, 1, 2], [.25, .75, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[-2, 98], [3.87344956, 103.873459]],
                    [[12.70884895, 112.70884705], [11.45574856, 111.45578003]]]
        interp = 'IDW'

        from niftynet.layer.resampler import SUPPORTED_BOUNDARY
        for b in list(SUPPORTED_BOUNDARY):
            self._test_partial_shape_correctness(
                input=self.get_3d_input2(False),
                rank=3,
                batch_size=2,
                grid=test_grid,
                interpolation=interp,
                boundary=b,
                expected_value=expected)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_3d_input2(False),
                    rank=3,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_3d_input2(False),
                    rank=-1,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)

    def test_2d_nearest_partial_shapes(self):
        test_grid = tf.constant(
            [[[.25, .25], [.25, .78]],
             [[.62, .25], [.25, .28]]], dtype=tf.float32)
        expected = [[[1, 2, -1],
                     [3, 4, -2]],
                    [[13, 14, -7],
                     [9, 10, -5]]]
        interp = 'nearest'

        from niftynet.layer.resampler import SUPPORTED_BOUNDARY
        for b in list(SUPPORTED_BOUNDARY):
            self._test_partial_shape_correctness(
                input=self.get_2d_input(False),
                rank=2,
                batch_size=2,
                grid=test_grid,
                interpolation=interp,
                boundary=b,
                expected_value=expected)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_2d_input(False),
                    rank=2,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_2d_input(False),
                    rank=-1,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)

    def test_3d_nearest_partial_shapes(self):
        test_grid = tf.constant(
            [[[0, 1, 2], [.25, .75, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[-2, 98], [3, 103]],
                    [[13, 113], [10, 110]]]
        interp = 'nearest'

        from niftynet.layer.resampler import SUPPORTED_BOUNDARY
        for b in list(SUPPORTED_BOUNDARY):
            self._test_partial_shape_correctness(
                input=self.get_3d_input2(False),
                rank=3,
                batch_size=2,
                grid=test_grid,
                interpolation=interp,
                boundary=b,
                expected_value=expected)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_3d_input2(False),
                    rank=3,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)

            with self.assertRaisesRegexp(TypeError, 'shape'):
                self._test_partial_shape_correctness(
                    input=self.get_3d_input2(False),
                    rank=-1,
                    batch_size=-1,
                    grid=test_grid,
                    interpolation=interp,
                    boundary=b,
                    expected_value=None)


if __name__ == "__main__":
    tf.test.main()
