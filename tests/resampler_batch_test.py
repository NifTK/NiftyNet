from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.layer.resampler import ResamplerLayer
from tests.niftynet_testcase import NiftyNetTestCase


class ResamplerTest(NiftyNetTestCase):

    def test_shape_interface(self):
        test_input = tf.zeros((2, 10, 10, 10, 3))
        test_coords = tf.zeros((3, 5, 5, 5, 3))
        # bad batch sizes
        with self.assertRaisesRegexp(ValueError, ''):
            out = ResamplerLayer()(test_input, test_coords)

        test_input = tf.zeros((2, 10, 10, 10, 3))
        test_coords = tf.zeros((5, 5, 5, 3))
        # bad batch sizes
        with self.assertRaisesRegexp(ValueError, ''):
            out = ResamplerLayer()(test_input, test_coords)

        test_input = tf.zeros((1, 10, 10, 3))
        test_coords = tf.zeros((1, 5, 5, 3))
        # bad n coordinates
        with self.assertRaisesRegexp(ValueError, ''):
            out = ResamplerLayer()(test_input, test_coords)

    def test_linear_shape(self):
        # 3D
        test_input = np.zeros((2, 8, 8, 8, 2))
        test_input[0, 0, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.ones((1, 5, 5, 5, 3)) * 0.1
        out = ResamplerLayer("LINEAR")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 0.9**3, atol=1e-5)))
            self.assertTrue(np.all(out_value[1, ...]==0))
            self.assertEqual(out_value.shape, (2, 5, 5, 5, 2))

        # 2D
        test_input = np.zeros((2, 8, 8, 2))
        test_input[0, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.ones((1, 5, 5, 2)) * 0.1
        out = ResamplerLayer("LINEAR")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(np.all(out_value[1, ...]==0))
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 0.9**2, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 5, 2))

        # 1D
        test_input = np.zeros((2, 8, 2))
        test_input[0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.ones((1, 5, 1)) * 0.1
        out = ResamplerLayer("LINEAR")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(np.all(out_value[1, ...]==0))
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 0.9, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 2))

    def test_linear_no_broadcasting(self):
        # 3D
        test_input = np.zeros((2, 8, 8, 8, 2))
        test_input[:, 0, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.concat([tf.ones((1, 5, 5, 5, 3)) * 0.1,
                                 tf.ones((1, 5, 5, 5, 3)) * 0.2], axis=0)
        out = ResamplerLayer("LINEAR")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 0.9**3, atol=1e-5)))
            self.assertTrue(
                np.all(np.isclose(out_value[1, ..., 0], 0.8**3, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 5, 5, 2))
        # 2D
        test_input = np.zeros((2, 8, 8, 2))
        test_input[:, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.concat([tf.ones((1, 5, 5, 2)) * 0.1,
                                 tf.ones((1, 5, 5, 2)) * 0.2], axis=0)
        out = ResamplerLayer("LINEAR")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 0.9**2, atol=1e-5)))
            self.assertTrue(
                np.all(np.isclose(out_value[1, ..., 0], 0.8**2, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 5, 2))
        # 1D
        test_input = np.zeros((2, 8, 2))
        test_input[:, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.concat([tf.ones((1, 5, 1)) * 0.1,
                                 tf.ones((1, 5, 1)) * 0.2], axis=0)
        out = ResamplerLayer("LINEAR")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 0.9, atol=1e-5)))
            self.assertTrue(
                np.all(np.isclose(out_value[1, ..., 0], 0.8, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 2))


    def test_nearest_shape(self):
        # 3D
        test_input = np.zeros((2, 8, 8, 8, 2))
        test_input[0, 0, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.ones((1, 5, 5, 5, 3)) * 0.1
        out = ResamplerLayer("NEAREST")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 1.0, atol=1e-5)))
            self.assertTrue(np.all(out_value[1, ...]==0))
            self.assertEqual(out_value.shape, (2, 5, 5, 5, 2))

        # 2D
        test_input = np.zeros((2, 8, 8, 2))
        test_input[0, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.ones((1, 5, 5, 2)) * 0.1
        out = ResamplerLayer("NEAREST")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(np.all(out_value[1, ...]==0))
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 1.0, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 5, 2))

        # 1D
        test_input = np.zeros((2, 8, 2))
        test_input[0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.ones((1, 5, 1)) * 0.1
        out = ResamplerLayer("NEAREST")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(np.all(out_value[1, ...]==0))
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 1.0, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 2))

    def test_nearest_no_broadcasting(self):
        # 3D
        test_input = np.zeros((2, 3, 3, 3, 2))
        test_input[:, 0, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.concat([tf.ones((1, 5, 5, 5, 3)) * 0.1,
                                 tf.ones((1, 5, 5, 5, 3)) * 1.2], axis=0)
        out = ResamplerLayer("NEAREST")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 1.0, atol=1e-5)))
            self.assertTrue(
                np.all(np.isclose(out_value[1, ..., 0], 0.0, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 5, 5, 2))
        # 2D
        test_input = np.zeros((2, 3, 3, 2))
        test_input[:, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.concat([tf.ones((1, 5, 5, 2)) * 0.1,
                                 tf.ones((1, 5, 5, 2)) * 1.2], axis=0)
        out = ResamplerLayer("NEAREST")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 1.0, atol=1e-5)))
            self.assertTrue(
                np.all(np.isclose(out_value[1, ..., 0], 0.0, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 5, 2))
        # 1D
        test_input = np.zeros((2, 3, 2))
        test_input[:, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.concat([tf.ones((1, 5, 1)) * 0.1,
                                 tf.ones((1, 5, 1)) * 1.2], axis=0)
        out = ResamplerLayer("NEAREST")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0], 1.0, atol=1e-5)))
            self.assertTrue(
                np.all(np.isclose(out_value[1, ..., 0], 0.0, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 2))

    def test_idw_shape(self):
        # 3D
        test_input = np.zeros((2, 8, 8, 8, 2))
        test_input[0, 0, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.ones((1, 5, 5, 5, 3)) * 0.1
        out = ResamplerLayer("IDW")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0],
                    1.0/(1. + 9./83 + 9./163 + 3./243), atol=1e-5)))
            self.assertTrue(np.all(out_value[1, ...]==0))
            self.assertEqual(out_value.shape, (2, 5, 5, 5, 2))

        # 2D
        test_input = np.zeros((2, 8, 8, 2))
        test_input[0, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.ones((1, 5, 5, 2)) * 0.1
        out = ResamplerLayer("IDW")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(np.all(out_value[1, ...]==0))
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0],
                    1./(2./41. + 1./81.0 + 1.0), atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 5, 2))

        # 1D
        test_input = np.zeros((2, 8, 2))
        test_input[0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.ones((1, 5, 1)) * 0.1
        out = ResamplerLayer("IDW")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(np.all(out_value[1, ...]==0))
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0],
                       100.0/(100.0+1/0.81), atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 2))

    def test_idw_no_broadcasting(self):
        # 3D
        test_input = np.zeros((2, 3, 3, 3, 2))
        test_input[:, 0, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.concat([tf.ones((1, 5, 5, 5, 3)) * 0.2,
                                 tf.ones((1, 5, 5, 5, 3)) * 1.2], axis=0)
        out = ResamplerLayer("IDW")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0],
                    1.0/(1. + 1./2. + 36./132. + 12./192.), atol=1e-5)))
            self.assertTrue(
                np.all(np.isclose(out_value[1, ..., 0], 0.0, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 5, 5, 2))
        # 2D
        test_input = np.zeros((2, 3, 3, 2))
        test_input[:, 0, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.concat([tf.ones((1, 5, 5, 2)) * 0.2,
                                 tf.ones((1, 5, 5, 2)) * 1.2], axis=0)
        out = ResamplerLayer("IDW")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0],
                    1.0/(1.0 + 1.0/16.0 + 16./68.), atol=1e-5)))
            self.assertTrue(
                np.all(np.isclose(out_value[1, ..., 0], 0.0, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 5, 2))
        # 1D
        test_input = np.zeros((2, 3, 2))
        test_input[:, 0, 0] = 1.0
        test_input = tf.constant(test_input)
        test_coords = tf.concat([tf.ones((1, 5, 1)) * 0.2,
                                 tf.ones((1, 5, 1)) * 1.2], axis=0)
        out = ResamplerLayer("IDW")(test_input, test_coords)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertTrue(
                np.all(np.isclose(out_value[0, ..., 0],
                    1.0/(1.0 + 1/16.0), atol=1e-5)))
            self.assertTrue(
                np.all(np.isclose(out_value[1, ..., 0], 0.0, atol=1e-5)))
            self.assertEqual(out_value.shape, (2, 5, 2))

if __name__ == "__main__":
    tf.test.main()
