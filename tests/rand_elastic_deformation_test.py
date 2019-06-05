from __future__ import absolute_import, print_function

import unittest
import numpy as np
import tensorflow as tf

from niftynet.layer.rand_elastic_deform import RandomElasticDeformationLayer
from niftynet.layer.rand_elastic_deform import sitk
from tests.niftynet_testcase import NiftyNetTestCase

SHAPE_3D = (10, 16, 2)
SHAPE_4D = (10, 16, 16, 2)
SHAPE_5D = (10, 32, 32, 8, 1)


@unittest.skipIf(not sitk, 'SimpleITK not found')
class RandDeformationTests(NiftyNetTestCase):
    def get_3d_input(self):
        input_3d = {'testdata': np.random.randn(*SHAPE_3D)}
        interp_order = {'testdata': (3,) * 2}
        return input_3d, interp_order

    def get_4d_input(self):
        input_4d = {'testdata': np.random.randn(*SHAPE_4D)}
        interp_order = {'testdata': (3,) * 2}
        return input_4d, interp_order

    def get_5d_input(self):
        input_5d = {'testdata': np.random.randn(*SHAPE_5D)}
        interp_order = {'testdata': (3,)}
        return input_5d, interp_order

    def test_4d_shape(self):
        x, interp_orders = self.get_4d_input()
        rand_deformation_layer = RandomElasticDeformationLayer(num_controlpoints=4,
                                                               std_deformation_sigma=15,
                                                               proportion_to_augment=0.5)
        rand_deformation_layer.randomise(x)
        out = rand_deformation_layer(x, interp_orders)
        self.assertEqual(out['testdata'].shape, SHAPE_4D)

    def test_5d_shape(self):
        x, interp_orders = self.get_5d_input()
        rand_deformation_layer = RandomElasticDeformationLayer(num_controlpoints=4,
                                                               std_deformation_sigma=15,
                                                               proportion_to_augment=0.5)
        rand_deformation_layer.randomise(x)
        out = rand_deformation_layer(x, interp_orders)
        self.assertEqual(out['testdata'].shape, SHAPE_5D)

    def test_no_deformation(self):
        # testing the 'proportion_to_augment' parameter
        rand_deformation_layer = RandomElasticDeformationLayer(num_controlpoints=4,
                                                               std_deformation_sigma=15,
                                                               proportion_to_augment=0.)
        for _ in range(100):
            x, interp_orders = self.get_5d_input()
            x_old = np.copy(x['testdata'])
            rand_deformation_layer.randomise(x)
            out = rand_deformation_layer(x, interp_orders)

            with self.cached_session():
                self.assertTrue(np.array_equal(out['testdata'], x_old))

    def test_deformation(self):
        # testing the 'proportion_to_augment' parameter

        rand_deformation_layer = RandomElasticDeformationLayer(num_controlpoints=4,
                                                               std_deformation_sigma=1,
                                                               proportion_to_augment=1.)
        for _ in range(100):
            x, interp_orders = self.get_5d_input()
            x_old = np.copy(x['testdata'])

            rand_deformation_layer.randomise(x)
            out = rand_deformation_layer(x, interp_orders)

            with self.cached_session():
                self.assertFalse(np.array_equal(out['testdata'], x_old))

    def test_deformation_on_2d_imgs(self):
        # testing the 'proportion_to_augment' parameter

        rand_deformation_layer = RandomElasticDeformationLayer(num_controlpoints=4,
                                                               std_deformation_sigma=1,
                                                               proportion_to_augment=1.,
                                                               spatial_rank = 2)
        for _ in range(100):
            x, interp_orders = self.get_3d_input()
            x_old = np.copy(x['testdata'])

            rand_deformation_layer.randomise(x)
            out = rand_deformation_layer(x, interp_orders)

            with self.cached_session():
                self.assertFalse(np.array_equal(out['testdata'], x_old))


if __name__ == "__main__":
    tf.test.main()
