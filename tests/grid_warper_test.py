from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.layer.grid_warper import AffineGridWarperLayer
from niftynet.layer.grid_warper import AffineWarpConstraints
from niftynet.layer.grid_warper import _create_affine_features
from tests.niftynet_testcase import NiftyNetTestCase

class GridWarperTest(NiftyNetTestCase):
    def test_regular_grids(self):
        out = _create_affine_features([3, 3], [2])
        expected = [
            np.array([0., 0., 0., 1., 1., 1., 2., 2., 2.], dtype=np.float32),
            np.array([0., 1., 2., 0., 1., 2., 0., 1., 2.], dtype=np.float32),
            np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=np.float32)]
        self.assertAllClose(expected, out)


class GridWarperRelativeTest(NiftyNetTestCase):
    def test_regular_grids(self):
        out = _create_affine_features([3, 3], [2], True)
        expected = [
            np.array([-1., -1., -1., 0., 0., 0., 1., 1., 1.], dtype=np.float32),
            np.array([-1., 0., 1., -1., 0., 1., -1., 0., 1.], dtype=np.float32),
            np.array([1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=np.float32)]
        self.assertAllClose(expected, out)


class AffineWarpConstraintsTest(NiftyNetTestCase):
    def test_default(self):
        aff_c = AffineWarpConstraints()
        self.assertAllEqual(aff_c.constraints,
                            ((None, None, None), (None, None, None)))
        self.assertAllClose(aff_c.num_free_params, 6)
        self.assertAllClose(aff_c.num_dim, 2)

        customise_constraints = (
            (None, None, 1, 1),
            (None, None, None, 2),
            (None, 1, None, 3))
        aff_c = AffineWarpConstraints(customise_constraints)
        self.assertEqual(aff_c.constraints, customise_constraints)

    def test_no_constraints(self):
        aff_c = AffineWarpConstraints.no_constraints(num_dim=2)
        self.assertEqual(aff_c.constraints,
                         ((None, None, None), (None, None, None)))

    def test_all_constraints_2d(self):
        test_x, test_y = 1, 2

        aff_c = AffineWarpConstraints.translation_2d(x=test_x, y=test_y)
        self.assertEqual(aff_c.constraints, ((None, None, 1), (None, None, 2)))

        aff_c = AffineWarpConstraints.scale_2d(x=test_x, y=test_y)
        self.assertEqual(aff_c.constraints, ((1, None, None), (None, 2, None)))

        aff_c = AffineWarpConstraints.shear_2d(x=test_x, y=test_y)
        self.assertEqual(aff_c.constraints, ((None, 1, None), (2, None, None)))

        aff_c = AffineWarpConstraints.no_shear_2d()
        self.assertEqual(aff_c.constraints, ((None, 0, None), (0, None, None)))

    def test_all_constraints_3d(self):
        test_x, test_y, test_z = 1, 2, 3

        aff_c = AffineWarpConstraints.translation_3d(
            x=test_x, y=test_y, z=test_z)
        self.assertEqual(
            aff_c.constraints,
            ((None, None, None, 1),
            (None, None, None, 2),
            (None, None, None, 3)))

        aff_c = AffineWarpConstraints.scale_3d(
            x=test_x, y=test_y, z=test_z)
        self.assertEqual(
            aff_c.constraints,
            ((1, None, None, None),
            (None, 2, None, None),
            (None, None, 3, None)))

        aff_c = AffineWarpConstraints.no_shear_3d()
        self.assertEqual(
            aff_c.constraints,
            ((None, 0, 0, None),
            (0, None, 0, None),
            (0, 0, None, None)))

    def test_combine_constraints(self):
        test_x, test_y, test_z = 1, 2, 3
        aff_c_1 = AffineWarpConstraints.translation_2d(x=test_x, y=test_y)
        aff_c_2 = AffineWarpConstraints.shear_2d(x=test_x, y=test_y)
        aff_comb = aff_c_1.combine_with(aff_c_2)
        self.assertEqual(aff_comb.constraints, ((None, 1, 1), (2, None, 2)))

        aff_c_1 = AffineWarpConstraints.translation_3d(
            x=test_x, y=test_y, z=test_z)
        aff_c_2 = AffineWarpConstraints.no_shear_3d()
        aff_comb = aff_c_1.combine_with(aff_c_2)
        self.assertEqual(aff_comb.constraints,
                         ((None, 0, 0, 1), (0, None, 0, 2), (0, 0, None, 3)))

        aff_c_1 = AffineWarpConstraints.translation_2d(x=test_x, y=test_y)
        aff_c_2 = AffineWarpConstraints.no_shear_3d()
        with self.assertRaisesRegexp(ValueError, ''):
            aff_c_1.combine_with(aff_c_2)


class AffineGridWarperLayerTest(NiftyNetTestCase):
    def _test_correctness(self, args, aff, expected_value):
        grid_warper = AffineGridWarperLayer(**args)
        computed_grid = grid_warper(aff)
        with self.cached_session() as sess:
            output_val = sess.run(computed_grid)
            self.assertAllClose(expected_value, output_val)

    def test_no_constraints(self):
        grid_warper = AffineGridWarperLayer(source_shape=(3, 3),
                                            output_shape=(2,))
        self.assertEqual(grid_warper.constraints.constraints,
                         ((None, None, None), (None, None, None)))

    def test_constraints(self):
        grid_warper = AffineGridWarperLayer(
            source_shape=(3, 3),
            output_shape=(2, 4),
            constraints=AffineWarpConstraints.no_shear_2d())
        self.assertEqual(grid_warper.constraints.constraints,
                         ((None, 0, None), (0, None, None)))

    def test_2d(self):
        aff = tf.constant([[1, 0, 5, 0, 1, 6]], dtype=tf.float32)
        params = {'source_shape': (3, 3), 'output_shape': (2, 2)}
        expected = [[[[5, 6], [5, 8]], [[7, 6], [7, 8]]]]
        self._test_correctness(params, aff, expected)

    def test_3d(self):
        aff = tf.constant([[
            2, 0, 0, 5,
            0, 2, 0, 6,
            0, 0, 1, 4]], dtype=tf.float32)
        params = {'source_shape': (3, 3, 3), 'output_shape': (2, 2, 2)}
        expected = [[[[[4, 5, 4], [4, 5, 6]], [[4, 9, 4], [4, 9, 6]]],
            [[[8, 5, 4], [8, 5, 6]], [[8, 9, 4], [8, 9, 6]]]]]
        self._test_correctness(params, aff, expected)

    def test_3d_2d(self):
        aff = tf.constant([[
            1.5, 0, 0, 3,
            0, 1.5, 0, 4,
            0, 0, 1.0, 5]], dtype=tf.float32)
        params = {'source_shape': (3, 3, 3), 'output_shape': (2, 2)}
        expected = [
            [[[2.5, 3.5, 5], [2.5, 6.5, 5]], [[5.5, 3.5, 5], [5.5, 6.5, 5]]]]
        self._test_correctness(params, aff, expected)

    def test_2d_1d(self):
        aff = tf.constant([[1.5, 2, 3, 2, 1.5, 4]], dtype=tf.float32)
        params = {'source_shape': (3, 3), 'output_shape': (2,)}
        expected = [[[0.5, 1.5], [3.5, 5.5]]]
        self._test_correctness(params, aff, expected)

    def test_3d_1d(self):
        aff = tf.constant([[
            1.5, 0, 0, 3,
            0, 2, 0, 4,
            0, 0, 1.0, 5]], dtype=tf.float32)
        params = {'source_shape': (3, 3, 3), 'output_shape': (2,)}
        expected = [[[2.5, 3, 5], [5.5, 3, 5]]]
        self._test_correctness(params, aff, expected)

    def test_3d_2d_scale(self):
        aff = tf.constant([[
            0, 0, 3,
            0, 0, 4,
            0, 0, 5]], dtype=tf.float32)
        params = {'source_shape': (3, 3, 3), 'output_shape': (2, 2),
            'constraints': AffineWarpConstraints.scale_3d(1, 1, 1)}
        expected = [[[[3, 4, 5], [3, 6, 5]], [[5, 4, 5], [5, 6, 5]]]]
        self._test_correctness(params, aff, expected)

    def test_3d_3d_translation(self):
        aff = tf.constant([[2, 0, 0,
            0, 1.5, 0,
            0, 0, 3]], dtype=tf.float32)
        params = {'source_shape': (3, 3, 3), 'output_shape': (2, 2),
            'constraints': AffineWarpConstraints.translation_3d(3, 4, 5)}
        expected = [[[[2, 3.5, 3], [2, 6.5, 3]], [[6, 3.5, 3], [6, 6.5, 3]]]]
        self._test_correctness(params, aff, expected)


class AffineGridWarperInvLayerTest(NiftyNetTestCase):
    def test_simple_inverse(self):
        expected_grid = np.array([[
            [[0.38, 0.08],
                [-0.02, 0.68],
                [-0.42, 1.28]],
            [[0.98, -0.32],
                [0.58, 0.28],
                [0.18, 0.88]],
            [[1.58, -0.72],
                [1.18, -0.12],
                [0.78, 0.48]]]], dtype=np.float32)

        inverse_grid = AffineGridWarperLayer(source_shape=(3, 3),
                                             output_shape=(2, 2)).inverse_op()
        aff = tf.constant([[1.5, 1.0, 0.2, 1.0, 1.5, 0.5]])
        output = inverse_grid(aff)
        with self.cached_session() as sess:
            out_val = sess.run(output)
        self.assertAllClose(out_val, expected_grid)


if __name__ == "__main__":
    tf.test.main()
