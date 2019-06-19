# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.linear_resize import LinearResizeLayer
from tests.niftynet_testcase import NiftyNetTestCase


class LinearResizeTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 4)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 4)
        x = tf.ones(input_shape)
        return x

    def run_test(self, new_size, expected_spatial_shape, is_3d=True):
        if is_3d:
            x = self.get_3d_input()
        else:
            x = self.get_2d_input()

        x_shape = x.shape.as_list()
        expected_shape = \
            [x_shape[0]] + list(expected_spatial_shape) + [x_shape[-1]]

        resize_layer = LinearResizeLayer(new_size=new_size)
        resized = resize_layer(x)
        print(resize_layer)
        with self.cached_session() as sess:
            out = sess.run(resized)
            self.assertAllClose(out.shape, expected_shape)

    def test_3d_shape(self):
        new_shape = (8, 8, 7)
        self.run_test(new_shape, new_shape)

        new_shape = (20, 18, 17)
        self.run_test(new_shape, new_shape)

        new_shape = (16, 16, 16)
        self.run_test(new_shape, new_shape)

    def test_2d_shape(self):
        new_shape = (8, 7)
        self.run_test(new_shape, new_shape, False)

        new_shape = (20, 18)
        self.run_test(new_shape, new_shape, False)

        new_shape = (16, 16)
        self.run_test(new_shape, new_shape, False)

    def test_int_shape(self):
        new_shape = 20

        self.run_test(new_shape, (new_shape,) * 2, False)
        self.run_test(new_shape, (new_shape,) * 3, True)

    def test_bad_int_shape(self):
        new_shape = 0

        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, ""):
            self.run_test(new_shape, (new_shape,) * 2, False)

        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, ""):
            self.run_test(new_shape, (new_shape,) * 3, True)

    def test_bad_shape(self):
        new_shape = (20, 5)
        with self.assertRaisesRegexp(AssertionError, ""):
            self.run_test(new_shape, new_shape)

        new_shape = (0, 0, 0)
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, ""):
            self.run_test(new_shape, new_shape)

    def test_ill_input(self):
        new_shape = (20, 15, 1)
        self.run_test(new_shape, new_shape[:2], False)

        new_shape = (20, 20)
        with self.assertRaisesRegexp(AssertionError, ""):
            self.run_test(new_shape, new_shape[:2], True)


if __name__ == "__main__":
    tf.test.main()
