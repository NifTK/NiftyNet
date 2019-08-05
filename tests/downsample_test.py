# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.downsample import DownSampleLayer
from tests.niftynet_testcase import NiftyNetTestCase

class DownSampleTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def _test_nd_downsample_output_shape(self,
                                         rank,
                                         param_dict,
                                         output_shape):
        if rank == 2:
            input_data = self.get_2d_input()
        elif rank == 3:
            input_data = self.get_3d_input()
        downsample_layer = DownSampleLayer(**param_dict)
        output_data = downsample_layer(input_data)
        print(downsample_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(output_data)
            self.assertAllClose(output_shape, out.shape)

    def test_3d_max_shape(self):
        input_param = {'func': 'MAX',
                       'kernel_size': 3,
                       'stride': 3}

        self._test_nd_downsample_output_shape(rank=3,
                                              param_dict=input_param,
                                              output_shape=(2, 6, 6, 6, 8))

    def test_3d_avg_shape(self):
        input_param = {'func': 'AVG',
                       'kernel_size': [3, 3, 2],
                       'stride': [3, 2, 1]}
        self._test_nd_downsample_output_shape(rank=3,
                                              param_dict=input_param,
                                              output_shape=(2, 6, 8, 16, 8))

    def test_3d_const_shape(self):
        input_param = {'func': 'CONSTANT',
                       'kernel_size': [1, 3, 2],
                       'stride': [3, 2, 2]}
        self._test_nd_downsample_output_shape(rank=3,
                                              param_dict=input_param,
                                              output_shape=(2, 6, 8, 8, 8))

    def test_2d_max_shape(self):
        input_param = {'func': 'CONSTANT',
                       'kernel_size': [1, 3],
                       'stride': 3}
        self._test_nd_downsample_output_shape(rank=2,
                                              param_dict=input_param,
                                              output_shape=(2, 6, 6, 8))

    def test_2d_avg_shape(self):
        input_param = {'func': 'AVG',
                       'kernel_size': [2, 3],
                       'stride': 2}
        self._test_nd_downsample_output_shape(rank=2,
                                              param_dict=input_param,
                                              output_shape=(2, 8, 8, 8))

    def test_2d_const_shape(self):
        input_param = {'func': 'CONSTANT',
                       'kernel_size': [2, 3],
                       'stride': [2, 3]}
        self._test_nd_downsample_output_shape(rank=2,
                                              param_dict=input_param,
                                              output_shape=(2, 8, 6, 8))


if __name__ == "__main__":
    tf.test.main()
