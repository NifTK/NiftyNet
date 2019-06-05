# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.residual_unit import ResidualUnit as Res
from niftynet.layer.residual_unit import SUPPORTED_OP as connection_types
from tests.niftynet_testcase import NiftyNetTestCase


class ResidualUnitTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def _test_nd_output_shape(self, rank, param_dict, expected_shape):
        if rank == 2:
            input_data = self.get_2d_input()
        elif rank == 3:
            input_data = self.get_3d_input()

        res_layer = Res(**param_dict)
        output_data = res_layer(input_data)
        print(res_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(output_data)
            self.assertAllClose(expected_shape, out.shape)

    def test_3d_shape(self):
        expected_shape = (2, 16, 16, 16, 1)
        self._test_nd_output_shape(3, {}, expected_shape)

        params = {'n_output_chns': 6, 'kernel_size': 5}
        expected_shape = (2, 16, 16, 16, 6)
        self._test_nd_output_shape(3, params, expected_shape)

        params = {'n_output_chns': 3, 'kernel_size': 3, 'acti_func': 'prelu'}
        expected_shape = (2, 16, 16, 16, 3)
        self._test_nd_output_shape(3, params, expected_shape)

        for type_str in connection_types:
            params = {'n_output_chns': 3, 'type_string': type_str}
            expected_shape = (2, 16, 16, 16, 3)
            self._test_nd_output_shape(3, params, expected_shape)

    def test_2d_shape(self):
        expected_shape = (2, 16, 16, 1)
        self._test_nd_output_shape(2, {}, expected_shape)

        params = {'n_output_chns': 16, 'kernel_size': 5, 'acti_func': 'relu'}
        expected_shape = (2, 16, 16, 16)
        self._test_nd_output_shape(2, params, expected_shape)

        params = {'n_output_chns': 3, 'kernel_size': 3, 'acti_func': 'prelu'}
        expected_shape = (2, 16, 16, 3)
        self._test_nd_output_shape(2, params, expected_shape)

        for type_str in connection_types:
            params = {'n_output_chns': 3, 'type_string': type_str}
            expected_shape = (2, 16, 16, 3)
            self._test_nd_output_shape(2, params, expected_shape)


if __name__ == "__main__":
    tf.test.main()
