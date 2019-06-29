# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.additive_upsample import ResidualUpsampleLayer
from tests.niftynet_testcase import NiftyNetTestCase

def get_3d_input():
    input_shape = (2, 16, 16, 16, 4)
    x = tf.ones(input_shape)
    return x

def get_2d_input():
    input_shape = (2, 16, 16, 4)
    x = tf.ones(input_shape)
    return x

class ResidualUpsampleTest(NiftyNetTestCase):
    def run_test(self, param_dict, expected_shape, is_3d=True):
        if is_3d:
            x = get_3d_input()
        else:
            x = get_2d_input()

        upsample_layer = ResidualUpsampleLayer(**param_dict)
        resized = upsample_layer(x)
        print(upsample_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(resized)
            self.assertAllClose(out.shape, expected_shape)

    def test_3d_shape(self):
        params = {'kernel_size': 3, 'stride': 2, 'n_splits': 2}
        expected_shape = (2, 32, 32, 32, 2)
        self.run_test(params, expected_shape, True)

        params = {'kernel_size': 2, 'stride': 3, 'n_splits': 4}
        expected_shape = (2, 48, 48, 48, 1)
        self.run_test(params, expected_shape, True)

        params = {'kernel_size': 2, 'stride': 3, 'n_splits': 1,
                  'acti_func': 'prelu'}
        expected_shape = (2, 48, 48, 48, 4)
        self.run_test(params, expected_shape, True)

        params = {'kernel_size': 2, 'stride': (3, 2, 3), 'n_splits': 1,
                  'acti_func': 'prelu'}
        expected_shape = (2, 48, 32, 48, 4)
        self.run_test(params, expected_shape, True)

    def test_2d_shape(self):
        params = {'kernel_size': 3, 'stride': 2, 'n_splits': 2}
        expected_shape = (2, 32, 32, 2)
        self.run_test(params, expected_shape, False)

        params = {'kernel_size': 2, 'stride': 3, 'n_splits': 4}
        expected_shape = (2, 48, 48, 1)
        self.run_test(params, expected_shape, False)

        params = {'kernel_size': 2, 'stride': 3, 'n_splits': 1,
                  'acti_func': 'prelu'}
        expected_shape = (2, 48, 48, 4)
        self.run_test(params, expected_shape, False)

        params = {'kernel_size': 2, 'stride': (3, 2), 'n_splits': 1,
                  'acti_func': 'prelu'}
        expected_shape = (2, 48, 32, 4)
        self.run_test(params, expected_shape, False)

    def test_float_params(self):
        params = {'kernel_size': 2.1, 'stride': 3, 'n_splits': 1.1,
                  'acti_func': 'prelu'}
        expected_shape = (2, 48, 48, 4)
        self.run_test(params, expected_shape, False)


    def test_bad_int_shape(self):
        params = {'kernel_size': 2, 'stride': 3, 'n_splits': 3,
                  'acti_func': 'prelu'}
        with self.assertRaisesRegexp(AssertionError, ""):
            self.run_test(params, (None,) * 2, False)

        with self.assertRaisesRegexp(AssertionError, ""):
            self.run_test(params, (None,) * 3, True)

if __name__ == "__main__":
    tf.test.main()
