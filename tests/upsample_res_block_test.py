# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.upsample_res_block import UpBlock
from tests.niftynet_testcase import NiftyNetTestCase

class UpsampleResBlockTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def _test_nd_output_shape(self,
                              rank,
                              param_dict,
                              output_shape):
        if rank == 2:
            input_data = self.get_2d_input()
        else:
            input_data = self.get_3d_input()

        upsample_layer = UpBlock(**param_dict)
        output_data = upsample_layer(input_data)
        print(upsample_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(output_data)
            self.assertAllClose(output_shape, out.shape)

    def test_3d_shape(self):
        expected_shape = (2, 32, 32, 32, 4)
        self._test_nd_output_shape(3, {}, expected_shape)

        params = {'n_output_chns': 2,
                  'kernel_size': 3, 'upsample_stride': 2,
                  'acti_func': 'relu'}
        expected_shape = (2, 32, 32, 32, 2)
        self._test_nd_output_shape(3, params, expected_shape)

        params = {'n_output_chns': 1,
                  'kernel_size': 4, 'upsample_stride': 3,
                  'acti_func': 'prelu'}
        expected_shape = (2, 48, 48, 48, 1)
        self._test_nd_output_shape(3, params, expected_shape)

        params = {'n_output_chns': 1,
                  'kernel_size': 4, 'upsample_stride': (3, 2, 3),
                  'acti_func': 'prelu'}
        expected_shape = (2, 48, 32, 48, 1)
        self._test_nd_output_shape(3, params, expected_shape)

        params = {'n_output_chns': 1,
                  'kernel_size': 4, 'upsample_stride': (3, 2, 3),
                  'acti_func': 'prelu', 'is_residual_upsampling': False}
        expected_shape = (2, 48, 32, 48, 1)
        self._test_nd_output_shape(3, params, expected_shape)

    def test_2d_shape(self):
        expected_shape = (2, 32, 32, 4)
        self._test_nd_output_shape(2, {}, expected_shape)

        params = {'n_output_chns': 2,
                  'kernel_size': 3, 'upsample_stride': 2,
                  'acti_func': 'relu'}
        expected_shape = (2, 32, 32, 2)
        self._test_nd_output_shape(2, params, expected_shape)

        params = {'n_output_chns': 1,
                  'kernel_size': 3, 'upsample_stride': 3,
                  'acti_func': 'prelu'}
        expected_shape = (2, 48, 48, 1)
        self._test_nd_output_shape(2, params, expected_shape)

        params = {'n_output_chns': 1,
                  'kernel_size': 3, 'upsample_stride': (3, 2),
                  'acti_func': 'prelu'}
        expected_shape = (2, 48, 32, 1)
        self._test_nd_output_shape(2, params, expected_shape)

        params = {'n_output_chns': 1,
                  'kernel_size': 3, 'upsample_stride': (3, 2),
                  'acti_func': 'prelu', 'is_residual_upsampling': False}
        expected_shape = (2, 48, 32, 1)
        self._test_nd_output_shape(2, params, expected_shape)

    def test_ill_params(self):
        params = {'n_output_chns': 16,
                  'kernel_size': 3, 'upsample_stride': 2,
                  'acti_func': 'relu'}
        expected_shape = (2, 32, 32, 32, 16)
        with self.assertRaisesRegexp(AssertionError, ''):
            self._test_nd_output_shape(3, params, expected_shape)

        params = {'n_output_chns': 2,
                  'kernel_size': 3, 'upsample_stride': 2,
                  'acti_func': 'relu', 'type_string': 'foo'}
        expected_shape = (2, 32, 32, 32, 16)
        with self.assertRaisesRegexp(ValueError, ''):
            self._test_nd_output_shape(3, params, expected_shape)


if __name__ == "__main__":
    tf.test.main()
