# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.layer.rand_flip import RandomFlipLayer
from tests.niftynet_testcase import NiftyNetTestCase

class RandFlipTest(NiftyNetTestCase):
    def test_1d_flip(self):
        a = np.array([[0, 1], [2, 3]])
        flip_layer = RandomFlipLayer(flip_axes=[0], flip_probability=1)
        flip_layer.randomise(spatial_rank=2)
        transformed_a = flip_layer._apply_transformation(a)
        with self.cached_session() as sess:
            self.assertTrue(
                np.array_equal(transformed_a, np.array([[2, 3], [0, 1]])))

    def test_no_flip(self):
        a = np.array([[0, 1], [2, 3]])
        flip_layer = RandomFlipLayer(flip_axes=[0], flip_probability=0)
        flip_layer.randomise(spatial_rank=2)
        transformed_a = flip_layer._apply_transformation(a)
        with self.cached_session() as sess:
            self.assertTrue(np.array_equal(transformed_a, a))

    def test_3d_flip(self):
        a = np.zeros(24).reshape(2, 3, 4)
        a[0, 0, 0] = 1
        flip_layer = RandomFlipLayer(flip_axes=[0, 1, 2], flip_probability=1)
        flip_layer.randomise(spatial_rank=3)
        transformed_a = flip_layer._apply_transformation(a)
        with self.cached_session() as sess:
            # cube of zeros with opposite corner as 1
            expected_a = np.zeros(24).reshape(2, 3, 4)
            expected_a[-1, -1, -1] = 1
            self.assertTrue(np.array_equal(transformed_a, expected_a))

    def test_no_flip_layer(self):
        a = np.array([[0, 1], [2, 3]])
        flip_layer = RandomFlipLayer(flip_axes=[0], flip_probability=0)
        flip_layer.randomise(spatial_rank=2)
        transformed_a = flip_layer(a)
        with self.cached_session() as sess:
            self.assertTrue(np.array_equal(transformed_a, a))

    def test_2d_flip_layer(self):
        a = np.array([[0, 1], [2, 3]])
        flip_layer = RandomFlipLayer(flip_axes=[0], flip_probability=1)
        flip_layer.randomise(spatial_rank=2)
        transformed_a = flip_layer(a)
        with self.cached_session() as sess:
            self.assertTrue(
                np.array_equal(transformed_a, np.array([[2, 3], [0, 1]])))

    def test_2d_flip_layer_1(self):
        a = np.array([[0, 1], [2, 3]])
        a = {'image': a}
        i = {'image': [0]}
        flip_layer = RandomFlipLayer(flip_axes=[0], flip_probability=1)
        flip_layer.randomise(spatial_rank=2)
        transformed_a = flip_layer(a, i)
        with self.cached_session() as sess:
            self.assertTrue(
                np.array_equal(transformed_a['image'],
                               np.array([[2, 3], [0, 1]])))


if __name__ == '__main__':
    tf.test.main()
