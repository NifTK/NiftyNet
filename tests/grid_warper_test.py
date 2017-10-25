from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.layer.grid_warper import AffineWarpConstraints
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


class AffineWarpConstraintsTest(tf.test.TestCase):
    def test_default(self):
        aff_c = AffineWarpConstraints()
        self.assertAllEqual(aff_c.constraints,
                            ((None, None, None), (None, None, None)))
        self.assertAllEqual(aff_c.mask,
                            ((True, True, True), (True, True, True)))
        self.assertAllClose(aff_c.num_free_params, 6)
        self.assertAllClose(aff_c.num_dim, 2)

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
            aff_comb = aff_c_1.combine_with(aff_c_2)


if __name__ == "__main__":
    tf.test.main()
