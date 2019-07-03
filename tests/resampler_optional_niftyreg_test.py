from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf
import tensorflow.test as tft

from niftynet.contrib.layer.resampler_optional_niftyreg import ResamplerOptionalNiftyRegLayer
from tests.niftynet_testcase import NiftyNetTestCase
import niftynet.contrib.layer.resampler_optional_niftyreg as resampler_module

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

    def _get_devs(self):
        devs = [False]
        if tft.is_gpu_available(cuda_only=True) and tft.is_built_with_cuda():
            devs += [True]

        return devs

    def _test_correctness(
            self, input, grid, interpolation, boundary, expected_value):
        resampler = ResamplerOptionalNiftyRegLayer(interpolation=interpolation,
                                                 boundary=boundary)
        out = resampler(input, grid)

        for use_gpu in self._get_devs():
            with self.cached_session(use_gpu=use_gpu) as sess:
                out_value = sess.run(out)
                self.assertAllClose(expected_value, out_value)

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

    def test_gradient_correctness(self):
        if not resampler_module.HAS_NIFTYREG_RESAMPLING:
            self.skipTest('Using native NiftyNet resampler; skipping test')
            return

        for inter in ('LINEAR', 'BSPLINE'):
            for b in ('ZERO', 'REPLICATE', 'SYMMETRIC'):
                for use_gpu in self._get_devs():
                    inputs = ((self.get_3d_input1(as_tensor=False),
                               [[[-5.2, .25, .25], [.25, .95, .25]],
                                [[.75, .25, .25], [.25, .25, .75]]]),
                              (self.get_2d_input(as_tensor=False),
                               [[[.25, .25], [.25, .78]],
                                [[.62, .25], [.25, .28]]]),)

                    for np_img, np_u in inputs:
                        with self.session(use_gpu=use_gpu):
                            np_u = np.array(np_u)

                            while len(np_u.shape) < len(np_img.shape):
                                np_u = np.expand_dims(np_u, axis=2)

                            img = tf.constant(np_img, dtype=tf.float32)
                            disp = tf.constant(np_u, dtype=tf.float32)

                            # multimodal needs addressing
                            if img.shape.as_list()[-1] > 1:
                                img = tf.reshape(img[...,0],
                                                 img.shape.as_list()[:-1] + [1])

                            warped = ResamplerOptionalNiftyRegLayer(interpolation=inter,
                                                                  boundary=b)
                            warped = warped(img, disp)
                            #warped = tf.reduce_sum(warped)

                            tgrad, refgrad = tft.compute_gradient(
                                disp,
                                disp.shape,
                                warped,
                                warped.shape)

                            error = np.power(tgrad - refgrad, 2).sum()
                            refmag = np.power(refgrad, 2).sum()

                            self.assertLessEqual(error, 1e-2*refmag)

    def test_image_derivative_correctness(self):
        if not resampler_module.HAS_NIFTYREG_RESAMPLING:
            self.skipTest('Using native NiftyNet resampler; skipping test')
            return

        for inter in ('LINEAR', 'BSPLINE'):
            for b in ('ZERO', 'REPLICATE', 'SYMMETRIC'):
                for use_gpu in self._get_devs():
                    if inter != 'LINEAR' and use_gpu:
                        continue

                    inputs = ((self.get_3d_input1(as_tensor=False),
                               [[[-5.2, .25, .25], [.25, .95, .25]],
                                [[.75, .25, .25], [.25, .25, .75]]]),
                              (self.get_2d_input(as_tensor=False),
                               [[[.25, .25], [.25, .78]],
                                [[.62, .25], [.25, .28]]]),)

                    for np_img, np_u in inputs:
                        with self.session(use_gpu=use_gpu):
                            np_u = np.array(np_u)

                            while len(np_u.shape) < len(np_img.shape):
                                np_u = np.expand_dims(np_u, axis=2)

                            img = tf.constant(np_img, dtype=tf.float32)
                            disp = tf.constant(np_u, dtype=tf.float32)

                            warped = ResamplerOptionalNiftyRegLayer(interpolation=inter,
                                                                  boundary=b)
                            warped = warped(img, disp)
                            #warped = tf.reduce_sum(warped)

                            tgrad, refgrad = tft.compute_gradient(
                                img,
                                img.shape,
                                warped,
                                warped.shape)

                            error = np.power(tgrad - refgrad, 2).sum()
                            refmag = np.power(refgrad, 2).sum()

                            self.assertLessEqual(error, 1e-2*refmag)

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

    def _test_partial_shape_correctness(self,
                                        input,
                                        rank,
                                        batch_size,
                                        grid,
                                        interpolation,
                                        boundary,
                                        expected_value=None):

        resampler = ResamplerOptionalNiftyRegLayer(interpolation=interpolation,
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

        for b in ('ZERO',):
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

        for b in ('ZERO',):
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

        for b in ('ZERO', 'REPLICATE'):
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

    def test_3d_nearest_partial_shapes(self):
        test_grid = tf.constant(
            [[[0, 1, 2], [.25, .75, .25]],
             [[.75, .25, .25], [.25, .25, .75]]],
            dtype=tf.float32)
        expected = [[[-2, 98], [3, 103]],
                    [[13, 113], [10, 110]]]
        interp = 'nearest'

        for b in ('ZERO', 'REPLICATE'):
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
