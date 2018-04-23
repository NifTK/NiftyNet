from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.contrib.layer.rand_elastic_deform import RandomElasticDeformationLayer


class RandDeformationTests(tf.test.TestCase):
    def get_4d_input(self):
        input_4d = {'testdata': np.random.randn(2, 16, 16, 8)}
        interp_order = {'testdata': (3,) * 8}
        return input_4d, interp_order

    def get_5d_input(self):
        input_5d = {'testdata': np.random.randn(2, 32, 32, 8, 1)}
        interp_order = {'testdata': (3,)}
        return input_5d, interp_order

    def test_4d_shape(self):
        x, interp_orders = self.get_4d_input()
        rand_deformation_layer = RandomElasticDeformationLayer(num_controlpoints=4,
                                                               std_deformation_sigma=15,
                                                               name='random_elastic_deformation',
                                                               proportion_to_augment=0.5)
        rand_deformation_layer.randomise(x)
        out = rand_deformation_layer(x, interp_orders)

    def test_5d_shape(self):
        x, interp_orders = self.get_5d_input()
        rand_deformation_layer = RandomElasticDeformationLayer(num_controlpoints=4,
                                                               std_deformation_sigma=15,
                                                               name='random_elastic_deformation',
                                                               proportion_to_augment=0.5)
        rand_deformation_layer.randomise(x)
        out = rand_deformation_layer(x, interp_orders)

    def test_no_deformation(self):
        # testing the 'proportion_to_augment' parameter
        rand_deformation_layer = RandomElasticDeformationLayer(num_controlpoints=4,
                                                               std_deformation_sigma=15,
                                                               name='random_elastic_deformation',
                                                               proportion_to_augment=0.)
        for _ in range(100):
            x, interp_orders = self.get_5d_input()
            x_old = np.copy(x['testdata'])
            rand_deformation_layer.randomise(x)
            out = rand_deformation_layer(x, interp_orders)

            with self.test_session():
                self.assertTrue(np.array_equal(out['testdata'], x_old))

    def test_deformation(self):
        # testing the 'proportion_to_augment' parameter

        rand_deformation_layer = RandomElasticDeformationLayer(num_controlpoints=4,
                                                               std_deformation_sigma=1,
                                                               name='random_elastic_deformation',
                                                               proportion_to_augment=1.)
        for _ in range(100):
            x, interp_orders = self.get_5d_input()
            x_old = np.copy(x['testdata'])

            rand_deformation_layer.randomise(x)
            out = rand_deformation_layer(x, interp_orders)

            with self.test_session():
                self.assertFalse(np.array_equal(out['testdata'], x_old))


if __name__ == "__main__":
    tf.test.main()
