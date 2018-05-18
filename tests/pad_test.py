# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.pad import PadLayer


class PaddingTest(tf.test.TestCase):
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


if __name__ == "__main__":
    tf.test.main()
