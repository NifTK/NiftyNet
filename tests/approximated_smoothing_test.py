# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.approximated_smoothing import SmoothingLayer as Smoothing


class SmoothingTest(tf.test.TestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def run_test(self, is_3d, sigma, type_str, expected_shape):
        if is_3d:
            x = self.get_3d_input()
        else:
            x = self.get_2d_input()

        smoothing_layer = Smoothing(sigma=sigma, type_str=type_str)
        smoothed = smoothing_layer(x)
        print(smoothing_layer)
        with self.test_session() as sess:
            out = sess.run(smoothed)
            #import matplotlib.pyplot as plt
            #if is_3d:
            #    plt.imshow(out[0,:,:,5,0])
            #else:
            #    plt.imshow(out[0,:,:,0])
            #plt.show()
            #self.assertAllClose(out.shape, expected_shape)

    def test_3d_inputs(self):
        self.run_test(True, 2, 'cauchy', [])


if __name__ == "__main__":
    tf.test.main()
