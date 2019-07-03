# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.approximated_smoothing import SmoothingLayer as Smoothing
from tests.niftynet_testcase import NiftyNetTestCase

class SmoothingTest(NiftyNetTestCase):
    def get_1d_input(self):
        input_shape = (2, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def run_test(self, ndim, sigma, type_str):
        if ndim == 3:
            x = self.get_3d_input()
        elif ndim == 2:
            x = self.get_2d_input()
        else:
            x = self.get_1d_input()

        smoothing_layer = Smoothing(sigma=sigma, type_str=type_str)
        smoothed = smoothing_layer(x)
        print(smoothing_layer)
        with self.cached_session() as sess:
            out = sess.run(smoothed)
            self.assertAllClose(out.shape, x.shape.as_list())
        return out

    def test_shape(self):
        self.run_test(1, 2, 'cauchy')
        self.run_test(1, 2, 'gaussian')
        self.run_test(2, 2, 'cauchy')
        self.run_test(2, 2, 'gaussian')
        self.run_test(3, 2, 'cauchy')
        self.run_test(3, 2, 'gaussian')

    def test_ill_inputs(self):
        self.run_test(3, -1, 'gaussian')
        self.run_test(3, -1, 'cauchy')
        self.run_test(3, 100, 'cauchy')

        with self.assertRaisesRegexp(ValueError, ''):
            self.run_test(3, -1, 'gassian')

    def test_original(self):
        out = self.run_test(3, 0, 'gaussian')
        self.assertTrue(np.all(out==1))
        out = self.run_test(2, 0, 'gaussian')
        self.assertTrue(np.all(out==1))
        out = self.run_test(1, 0, 'gaussian')
        self.assertTrue(np.all(out==1))
        out = self.run_test(3, 0, 'cauchy')
        self.assertTrue(np.all(out==1))
        out = self.run_test(2, 0, 'cauchy')
        self.assertTrue(np.all(out==1))
        out = self.run_test(1, 0, 'cauchy')
        self.assertTrue(np.all(out==1))


if __name__ == "__main__":
    tf.test.main()
