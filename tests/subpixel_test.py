from __future__ import division, absolute_import, print_function

import functools as ft
import numpy as np
import tensorflow as tf

from niftynet.layer.subpixel import SubPixelLayer

class SubPixelTest(tf.test.TestCase):
    """
    Test for niftynet.layer.subpixel.SubPixelLayer.
    Mostly adapted from convolution_test.py
    """

    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x_3d = tf.ones(input_shape)
        return x_3d

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x_2d = tf.ones(input_shape)
        return x_2d

    def _test_subpixel_output_shape(self,
                                    input_data,
                                    param_dict,
                                    output_shape):

        layer = SubPixelLayer(**param_dict)
        output_data = layer(input_data)
        print(layer)
        with self.session() as sess:
            sess.run(tf.global_variables_initializer())
            output_value = sess.run(output_data)
            self.assertAllClose(output_shape, output_value.shape)

    def _make_output_shape(self, data, upsampling):
        data_shape = data.shape.as_list()
        output_shape = [data_shape[0]]
        output_shape += [upsampling*d for d in data_shape[1:-1]]
        output_shape += [data_shape[-1]]

        return output_shape

    def test_3d_default(self):
        data = self.get_3d_input()

        output_shape = self._make_output_shape(data, 3)

        self._test_subpixel_output_shape(data,
                                         {},
                                         output_shape)


    def test_3d_bespoke(self):
        data = self.get_3d_input()
        upsampling = 4

        output_shape = self._make_output_shape(data, upsampling)

        params = {'upsample_factor': upsampling,
                  'layer_configurations': ((6, 32),
                                           (3, 32),
                                           (2, 16),
                                           (4, -1)),
                  'padding': 'SAME'}

        self._test_subpixel_output_shape(data,
                                         params,
                                         output_shape)



    def test_2d_bespoke(self):
        data = self.get_2d_input()
        upsampling = 6

        output_shape = self._make_output_shape(data, upsampling)

        params = {'upsample_factor': upsampling,
                  'layer_configurations': ((6, 32),
                                           (3, 32),
                                           (2, 16),
                                           (4, -1)),
                  'padding': 'SAME'}

        self._test_subpixel_output_shape(data,
                                         params,
                                         output_shape)


if __name__ == "__main__":
    tf.test.main()
