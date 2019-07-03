# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.additive_upsample import AdditiveUpsampleLayer
from tests.niftynet_testcase import NiftyNetTestCase

def get_3d_input():
    input_shape = (2, 16, 16, 16, 4)
    x = tf.ones(input_shape)
    return x

def get_2d_input():
    input_shape = (2, 16, 16, 4)
    x = tf.ones(input_shape)
    return x

class AdditiveUpsampleTest(NiftyNetTestCase):
    def run_test(self, new_size, n_splits, expected_shape, is_3d=True):
        if is_3d:
            x = get_3d_input()
        else:
            x = get_2d_input()

        resize_layer = AdditiveUpsampleLayer(
            new_size=new_size, n_splits=n_splits)
        resized = resize_layer(x)
        print(resize_layer)
        with self.cached_session() as sess:
            out = sess.run(resized)
            self.assertAllClose(out.shape, expected_shape)

    def test_3d_shape(self):
        new_shape = (8, 8, 7)
        n_splits = 2
        expected_shape = (2,) + new_shape + (2,)
        self.run_test(new_shape, n_splits, expected_shape, True)

        new_shape = (20, 18, 17)
        n_splits = 2
        expected_shape = (2,) + new_shape + (2,)
        self.run_test(new_shape, n_splits, expected_shape, True)

        new_shape = (16, 16, 16)
        n_splits = 2
        expected_shape = (2,) + new_shape + (2,)
        self.run_test(new_shape, n_splits, expected_shape)

        new_shape = (16, 16, 16)
        n_splits = 4
        expected_shape = (2,) + new_shape + (1,)
        self.run_test(new_shape, n_splits, expected_shape)

    def test_2d_shape(self):
        new_shape = (8, 7)
        n_splits = 2
        expected_shape = (2,) + new_shape + (2,)
        self.run_test(new_shape, n_splits, expected_shape, False)

        new_shape = (20, 18)
        n_splits = 2
        expected_shape = (2,) + new_shape + (2,)
        self.run_test(new_shape, n_splits, expected_shape, False)

        new_shape = (16, 16)
        n_splits = 2
        expected_shape = (2,) + new_shape + (2,)
        self.run_test(new_shape, n_splits, expected_shape, False)

        new_shape = (16, 16)
        n_splits = 4
        expected_shape = (2,) + new_shape + (1,)
        self.run_test(new_shape, n_splits, expected_shape, False)

    def test_int_shape(self):
        new_shape = 20
        n_splits = 2
        expected_shape = (2,) + (new_shape,) * 2 + (2,)
        self.run_test(new_shape, n_splits, expected_shape, False)
        expected_shape = (2,) + (new_shape,) * 3 + (2,)
        self.run_test(new_shape, n_splits, expected_shape, True)

        n_splits = 4
        expected_shape = (2,) + (new_shape,) * 2 + (1,)
        self.run_test(new_shape, n_splits, expected_shape, False)
        expected_shape = (2,) + (new_shape,) * 3 + (1,)
        self.run_test(new_shape, n_splits, expected_shape, True)

    def test_bad_int_shape(self):
        new_shape = 0
        n_splits = 2
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, ""):
            self.run_test(new_shape, n_splits, (new_shape,) * 2, False)

        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, ""):
            self.run_test(new_shape, n_splits, (new_shape,) * 3, True)

        new_shape = 2
        n_splits = 0
        with self.assertRaisesRegexp(AssertionError, ""):
            self.run_test(new_shape, n_splits, (new_shape,) * 2, False)

        with self.assertRaisesRegexp(AssertionError, ""):
            self.run_test(new_shape, n_splits, (new_shape,) * 3, True)

    def test_bad_shape(self):
        new_shape = (20, 5)
        n_splits = 2
        with self.assertRaisesRegexp(AssertionError, ""):
            self.run_test(new_shape, n_splits, new_shape)

        new_shape = (0, 0, 0)
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, ""):
            self.run_test(new_shape, n_splits, new_shape)

    def test_ill_input(self):
        new_shape = (20, 15, 1)
        n_splits = 2
        expected_shape = (2,) + new_shape[:2] + (2,)
        self.run_test(new_shape, n_splits, expected_shape, False)

        new_shape = (20, 20)
        n_splits = 2
        expected_shape = (2,) + new_shape[:2] + (2,)
        with self.assertRaisesRegexp(AssertionError, ""):
            self.run_test(new_shape, n_splits, expected_shape, True)

if __name__ == "__main__":
    tf.test.main()
