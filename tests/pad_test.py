# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.pad import PadLayer
from tests.niftynet_testcase import NiftyNetTestCase


class PaddingTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (16, 16, 16, 8)
        x = np.ones(input_shape)
        return x

    def get_3d_input_dict(self):
        input_shape = (13, 14, 10, 8)
        x = {'image': np.ones(input_shape)}
        return x

    def run_test(self, is_array, layer_param, expected_shape):
        if is_array:
            x = self.get_3d_input()
        else:
            x = self.get_3d_input_dict()
        padding_layer = PadLayer(**layer_param)
        out_acti, _ = padding_layer(x)
        print(padding_layer)
        if is_array:
            self.assertAllClose(out_acti.shape, expected_shape)
        else:
            self.assertAllClose(out_acti['image'].shape, expected_shape)

    def run_inverse_test(self, is_array, layer_param, expected_shape):
        if is_array:
            x = self.get_3d_input()
        else:
            x = self.get_3d_input_dict()
        padding_layer = PadLayer(**layer_param)
        out_acti, _ = padding_layer.inverse_op(x)
        print(padding_layer)
        if is_array:
            self.assertAllClose(out_acti.shape, expected_shape)
        else:
            self.assertAllClose(out_acti['image'].shape, expected_shape)

    # 3d test
    def test_3d_pad_shape(self):
        input_param = {'image_name': ('image',),
                       'border': (3,)}
        self.run_test(True, input_param, (22, 16, 16, 8))
        self.run_inverse_test(True, input_param, (10, 16, 16, 8))

    def test_3d_twopad_shape(self):
        input_param = {'image_name': ('image',),
                       'border': (3, 3)}
        self.run_test(True, input_param, (22, 22, 16, 8))
        self.run_inverse_test(True, input_param, (10, 10, 16, 8))

    def test_3d_int_pad_shape(self):
        input_param = {'image_name': ('image',),
                       'border': 3}
        with self.assertRaisesRegexp(TypeError, 'iter'):
            self.run_test(True, input_param, (22, 22, 16, 8))
        with self.assertRaisesRegexp(TypeError, 'iter'):
            self.run_inverse_test(True, input_param, (10, 10, 16, 8))

    def test_3d_large_pad_shape(self):
        input_param = {'image_name': ('image',),
                       'border': (18,)}
        self.run_test(True, input_param, (52, 16, 16, 8))
        self.run_inverse_test(True, input_param, (16, 16, 16, 8))

    # 3d dict test
    def test_3d_dict_pad_shape(self):
        input_param = {'image_name': ('image',),
                       'border': (3,)}
        self.run_test(False, input_param, (19, 14, 10, 8))
        self.run_inverse_test(False, input_param, (7, 14, 10, 8))

    def test_3d_dict_twopad_shape(self):
        input_param = {'image_name': ('image',),
                       'border': (3, 3)}
        self.run_test(False, input_param, (19, 20, 10, 8))
        self.run_inverse_test(False, input_param, (7, 8, 10, 8))

    def test_3d_dict_int_pad_shape(self):
        input_param = {'image_name': ('image',),
                       'border': 3}
        with self.assertRaisesRegexp(TypeError, 'iter'):
            self.run_test(False, input_param, (22, 22, 16, 8))
        with self.assertRaisesRegexp(TypeError, 'iter'):
            self.run_inverse_test(False, input_param, (10, 10, 16, 8))

    def test_3d_dict_large_pad_shape(self):
        input_param = {'image_name': ('image',),
                       'border': (18, 2, 1)}
        self.run_test(False, input_param, (49, 18, 12, 8))
        self.run_inverse_test(False, input_param, (13, 10, 8, 8))

    def test_pad_to_simple(self):
        rand_image = np.random.random([10, 10, 2, 1])
        data_dict = {'image': rand_image}
        tst = PadLayer(('image',), (0,), pad_to=(52, 52, 2))

        padded = tst.layer_op(data_dict)
        self.assertTrue(padded[0]['image'].shape == (52, 52, 2, 1))
        depadded = tst.inverse_op(padded[0])
        self.assertTrue(np.all(depadded[0]['image'] == rand_image))

    def test_pad_to_smaller_window_than_input(self):
        rand_image = np.random.random([10, 10, 2, 1])
        data_dict = {'image': rand_image}
        tst = PadLayer(('image',), (0,), pad_to=(5, 5, 10))
        # test straightforward pad_to
        padded = tst.layer_op(data_dict)
        self.assertTrue(padded[0]['image'].shape == (10, 10, 10, 1))
        depadded = tst.inverse_op(padded[0])
        self.assertTrue(np.all(depadded[0]['image'] == rand_image))

    def test_pad_to_odd_numbers(self):
        rand_image = np.random.random([10, 10, 2, 1])
        data_dict = {'image': rand_image}
        tst = PadLayer(('image',), (0,), pad_to=(15, 17, 10))
        # test straightforward pad_to
        padded = tst.layer_op(data_dict)
        self.assertTrue(padded[0]['image'].shape == (15, 17, 10, 1))
        depadded = tst.inverse_op(padded[0])
        self.assertTrue(np.all(depadded[0]['image'] == rand_image))

    def test_pad_to_without_data_dict(self):
        rand_image = np.random.random([10, 10, 2, 1])
        data_dict = {'image': rand_image}
        # test without dictionary
        tst = PadLayer(('image',), (0,), pad_to=(5, 5, 10))
        padded = tst.layer_op(data_dict['image'])[0]
        self.assertTrue(padded.shape == (10, 10, 10, 1))
        depadded = tst.inverse_op(padded)[0]
        self.assertTrue(np.all(depadded == rand_image))


if __name__ == "__main__":
    tf.test.main()
