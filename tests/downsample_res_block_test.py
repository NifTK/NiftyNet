# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.downsample_res_block import DownBlock
from tests.niftynet_testcase import NiftyNetTestCase


class DownsampleResBlockTest(NiftyNetTestCase):
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
        elif rank == 3:
            input_data = self.get_3d_input()

        downsample_layer = DownBlock(**param_dict)
        output_data = downsample_layer(input_data)
        print(downsample_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(output_data)
            self.assertAllClose(output_shape, out[0].shape)

    def test_3d_shape(self):
        expected_shape = (2, 8, 8, 8, 4)
        self._test_nd_output_shape(3, {}, expected_shape)

        params = {'n_output_chns': 16, 'kernel_size': 5,
                  'downsample_kernel_size': 3, 'downsample_stride': 2,
                  'acti_func': 'relu'}
        expected_shape = (2, 8, 8, 8, 16)
        self._test_nd_output_shape(3, params, expected_shape)

        params = {'n_output_chns': 16, 'kernel_size': 3,
                  'downsample_kernel_size': 4, 'downsample_stride': 3,
                  'acti_func': 'prelu'}
        expected_shape = (2, 6, 6, 6, 16)
        self._test_nd_output_shape(3, params, expected_shape)

    def test_2d_shape(self):
        expected_shape = (2, 8, 8, 4)
        self._test_nd_output_shape(2, {}, expected_shape)

        params = {'n_output_chns': 16, 'kernel_size': 5,
                  'downsample_kernel_size': 3, 'downsample_stride': 2,
                  'acti_func': 'relu'}
        expected_shape = (2, 8, 8, 16)
        self._test_nd_output_shape(2, params, expected_shape)

        params = {'n_output_chns': 16, 'kernel_size': 3,
                  'downsample_kernel_size': 4, 'downsample_stride': 3,
                  'acti_func': 'prelu'}
        expected_shape = (2, 6, 6, 16)
        self._test_nd_output_shape(2, params, expected_shape)



if __name__ == "__main__":
    tf.test.main()
