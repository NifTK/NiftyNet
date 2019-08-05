# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.spatial_gradient import SpatialGradientLayer
from tests.niftynet_testcase import NiftyNetTestCase

class SpatialGradientTest(NiftyNetTestCase):

    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def _test_nd_gradient_output_shape(self,
                                       rank,
                                       param_dict,
                                       expected_shape,
                                       expected_value=None):
        if rank == 2:
            input_data = self.get_2d_input()
        elif rank == 3:
            input_data = self.get_3d_input()

        gradient_layer = SpatialGradientLayer(**param_dict)
        output_data = gradient_layer(input_data)
        print(gradient_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(output_data)
            if expected_value is not None:
                self.assertAllClose(expected_value, out)
            self.assertAllClose(expected_shape, out.shape)

    def test_cropping_shape(self):
        for spatial_ind in [-1, 0, 1]:
            input_param = {'spatial_axis': spatial_ind, 'do_cropping': True}

            expected_shape = (2, 14, 14, 14, 8)
            expected_value = np.zeros(expected_shape)
            self._test_nd_gradient_output_shape(
                rank=3, param_dict=input_param,
                expected_shape=expected_shape,
                expected_value=expected_value)

            expected_shape = (2, 14, 14, 8)
            expected_value = np.zeros(expected_shape)
            self._test_nd_gradient_output_shape(
                rank=2, param_dict=input_param,
                expected_shape=expected_shape,
                expected_value=expected_value)

    def test_no_cropping_shape(self):
        input_param = {'spatial_axis': 0, 'do_cropping': False}
        expected_shape = (2, 14, 16, 16, 8)
        expected_value = np.zeros(expected_shape)
        self._test_nd_gradient_output_shape(
            rank=3, param_dict=input_param,
            expected_shape=expected_shape,
            expected_value=expected_value)

        input_param = {'spatial_axis': 0, 'do_cropping': False}
        expected_shape = (2, 14, 16, 8)
        expected_value = np.zeros(expected_shape)
        self._test_nd_gradient_output_shape(
            rank=2, param_dict=input_param,
            expected_shape=expected_shape,
            expected_value=expected_value)

        input_param = {'spatial_axis': 1, 'do_cropping': False}
        expected_shape = (2, 16, 14, 16, 8)
        expected_value = np.zeros(expected_shape)
        self._test_nd_gradient_output_shape(
            rank=3, param_dict=input_param,
            expected_shape=expected_shape,
            expected_value=expected_value)

        input_param = {'spatial_axis': 1, 'do_cropping': False}
        expected_shape = (2, 16, 14, 8)
        expected_value = np.zeros(expected_shape)
        self._test_nd_gradient_output_shape(
            rank=2, param_dict=input_param,
            expected_shape=expected_shape,
            expected_value=expected_value)

        input_param = {'spatial_axis': 2, 'do_cropping': False}
        expected_shape = (2, 16, 16, 14, 8)
        expected_value = np.zeros(expected_shape)
        self._test_nd_gradient_output_shape(
            rank=3, param_dict=input_param,
            expected_shape=expected_shape,
            expected_value=expected_value)

        input_param = {'spatial_axis': -1, 'do_cropping': False}
        self._test_nd_gradient_output_shape(
            rank=3, param_dict=input_param,
            expected_shape=expected_shape,
            expected_value=expected_value)


if __name__ == "__main__":
    tf.test.main()
