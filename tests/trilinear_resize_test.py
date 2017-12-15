# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.trilinear_resize import TrilinearResizeLayer


class TrilinearResizeTest(tf.test.TestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 4)
        x = tf.ones(input_shape)
        return x

    def run_test(self, size_3d, expected_spatial_shape):
        x = self.get_3d_input()
        x_shape = x.get_shape().as_list()
        expected_shape = \
            [x_shape[0]] + list(expected_spatial_shape) + [x_shape[-1]]

        resize_layer = TrilinearResizeLayer(size_3d=size_3d)
        resized = resize_layer(x)
        print(resize_layer)
        with self.test_session() as sess:
            out = sess.run(resized)
            self.assertAllClose(out.shape, expected_shape)

    def test_3d_shape(self):
        new_shape = (8, 8, 7)
        self.run_test(new_shape, new_shape)

        new_shape = (20, 18, 17)
        self.run_test(new_shape, new_shape)

        new_shape = (16, 16, 16)
        self.run_test(new_shape, new_shape)

    def test_bad_shape(self):
        new_shape = (20, 5)
        with self.assertRaisesRegexp(AssertionError, ""):
            self.run_test(new_shape, new_shape)

        new_shape = (0, 0, 0)
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, ""):
            self.run_test(new_shape, new_shape)

    def test_bad_input(self):
        x = tf.ones((2, 10, 10, 3))
        resize_layer = TrilinearResizeLayer(size_3d=(1, 1, 1))
        with self.assertRaisesRegexp(AssertionError, ""):
            resize_layer(x)



if __name__ == "__main__":
    tf.test.main()
