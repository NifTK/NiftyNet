from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.rand_bias_field import RandomBiasFieldLayer
from tests.niftynet_testcase import NiftyNetTestCase

SHAPE_4D = (10, 16, 16, 2)
SHAPE_5D = (10, 32, 32, 8, 1)


class RandDeformationTests(NiftyNetTestCase):
    # def get_3d_input(self):
    #     input_3d = {'image': np.random.randn(10, 16, 2)}
    #     interp_order = {'image': (3,) * 2}
    #     return input_3d, interp_order

    def get_4d_input(self):
        input_4d = {'image': np.random.randn(*SHAPE_4D)}
        interp_order = {'image': (3,) * 2}
        return input_4d, interp_order

    def get_5d_input(self):
        input_5d = {'image': np.random.randn(*SHAPE_5D)}
        interp_order = {'image': (3,)}
        return input_5d, interp_order

    def test_4d_shape(self):
        x, interp_orders = self.get_4d_input()
        rand_bias_field_layer = RandomBiasFieldLayer()
        rand_bias_field_layer.randomise()
        out = rand_bias_field_layer(x, interp_orders)
        self.assertEqual(out['image'].shape, SHAPE_4D)

    def test_5d_shape(self):
        x, interp_orders = self.get_5d_input()
        rand_bias_field_layer = RandomBiasFieldLayer()
        rand_bias_field_layer.randomise()
        out = rand_bias_field_layer(x, interp_orders)
        self.assertEqual(out['image'].shape, SHAPE_5D)

    def test_augmentation(self):
        rand_bias_field_layer = RandomBiasFieldLayer()
        x, interp_orders = self.get_5d_input()
        x_old = np.copy(x['image'])
        for _ in range(10):
            x['image'] = np.copy(x_old)

            rand_bias_field_layer.randomise()
            out = rand_bias_field_layer(x, interp_orders)
            out = np.copy(out['image'])

            rand_bias_field_layer.init_order(2)
            rand_bias_field_layer.init_uniform_coeff((-5.0, 5.0))

            rand_bias_field_layer.randomise()
            x['image'] = np.copy(x_old)
            out2 = rand_bias_field_layer(x, interp_orders)

            self.assertFalse(np.array_equal(out, x_old))
            self.assertFalse(np.array_equal(out2['image'], x_old))
            self.assertFalse(np.array_equal(out, out2['image']))

    # def test_gugmentation_on_2d_imgs(self):
    #     rand_bias_field_layer = RandomBiasFieldLayer()
    #     for _ in range(100):
    #         x, interp_orders = self.get_3d_input()
    #         x_old = np.copy(x['image'])

    #         rand_bias_field_layer.randomise(x)
    #         out = rand_bias_field_layer(x, interp_orders)

    #         with self.cached_session():
    #             self.assertFalse(np.array_equal(out['image'], x_old))


if __name__ == "__main__":
    tf.test.main()
