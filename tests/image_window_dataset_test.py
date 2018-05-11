# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.io.image_reader import ImageReader
from niftynet.contrib.dataset_sampler.image_window_dataset import \
    ImageWindowDataset


IMAGE_PATH_2D_1 = os.path.join('.', 'example_volumes', 'gan_test_data')
IMAGE_PATH_3D = os.path.join('.', 'testing_data')

def get_2d_reader():
    data_param = {'mr': {'path_to_search': IMAGE_PATH_2D_1}}
    reader = ImageReader().initialise(data_param)
    return reader

def get_3d_reader():
    data_param = {'mr': {'path_to_search': IMAGE_PATH_3D,
                         'filename_contains': 'FLAIR',
                         'interp_order': 1}}
    reader = ImageReader().initialise(data_param)
    return reader


class ImageWindowDataset_2D_Test(tf.test.TestCase):
    def assert_window(self, window):
        self.assertEqual(window['mr'].shape[1:3], (120, 160))
        self.assertEqual(window['mr_location'][0, 1:].tolist(),
                         [0, 0, 0, 120, 160, 1])
        self.assertEqual(window['mr'].dtype, np.float32)
        self.assertEqual(window['mr_location'].dtype, np.int32)

    def assert_tf_window(self, sampler):
        with self.test_session() as sess:
            sampler.run_threads(sess)
            window = sess.run(sampler.pop_batch_op())
        self.assert_window(window)

    def test_simple(self):
        sampler = ImageWindowDataset(reader=get_2d_reader())
        self.assert_tf_window(sampler)
        self.assert_window(sampler())

    def test_batch_size(self):
        # batch size doesn't change the numpy interface
        sampler = ImageWindowDataset(reader=get_2d_reader(), batch_size=2)
        self.assert_tf_window(sampler)
        self.assert_window(sampler())

    def test_window_size(self):
        sampler = ImageWindowDataset(reader=get_2d_reader(),
                                     window_sizes=(0, 0, 0), batch_size=2)
        self.assert_tf_window(sampler)
        self.assert_window(sampler())

    def test_window_size_dict(self):
        sampler = ImageWindowDataset(reader=get_2d_reader(),
                                     window_sizes={'mr': (0, 0, 0)},
                                     batch_size=2)
        self.assert_tf_window(sampler)
        self.assert_window(sampler())

    # sampler layer_op()'s output shape is not checked
    #def test_wrong_window_size_dict(self):
    #    sampler = ImageWindowDataset(reader=get_2d_reader(),
    #                                 batch_size=2,
    #                                 window_sizes=(3,3,0))
    #    self.assert_tf_window(sampler)

    def test_windows_per_image(self):
        with self.assertRaisesRegexp(AssertionError, ''):
            sampler = ImageWindowDataset(reader=get_2d_reader(), batch_size=2,
                                         windows_per_image=2)
            self.assert_window(sampler())

    def test_epoch(self):
        reader = get_2d_reader()
        batch_size = 3
        sampler = ImageWindowDataset(reader=reader, batch_size=batch_size,
                                     epoch=1)
        with self.test_session() as sess:
            sampler.run_threads(sess)
            iters = 0
            try:
                for _ in range(400):
                    window = sess.run(sampler.pop_batch_op())
                    iters = iters + 1
            except tf.errors.OutOfRangeError:
                pass
            # batch size 2, 40 images in total
            self.assertEqual(np.ceil(reader.num_subjects/np.float(batch_size)),
                             iters)


class ImageWindowDataset_3D_Test(tf.test.TestCase):
    def assert_window(self, window):
        self.assertEqual(window['mr'].shape[1:4], (256, 168, 256))
        self.assertEqual(window['mr_location'][0, 1:].tolist(),
                         [0, 0, 0, 256, 168, 256])
        self.assertEqual(window['mr'].dtype, np.float32)
        self.assertEqual(window['mr_location'].dtype, np.int32)

    def assert_tf_window(self, sampler):
        with self.test_session() as sess:
            sampler.run_threads(sess)
            window = sess.run(sampler.pop_batch_op())
        self.assert_window(window)

    def test_simple(self):
        sampler = ImageWindowDataset(reader=get_3d_reader())
        self.assert_tf_window(sampler)
        self.assert_window(sampler())

    def test_batch_size(self):
        # batch size doesn't change the numpy interface
        sampler = ImageWindowDataset(reader=get_3d_reader(), batch_size=2)
        self.assert_tf_window(sampler)
        self.assert_window(sampler())

    def test_window_size(self):
        sampler = ImageWindowDataset(reader=get_3d_reader(),
                                     window_sizes=(0, 0, 0), batch_size=2)
        self.assert_tf_window(sampler)
        self.assert_window(sampler())

    def test_window_size_dict(self):
        sampler = ImageWindowDataset(reader=get_3d_reader(),
                                     window_sizes={'mr': (0, 0, 0)},
                                     batch_size=2)
        self.assert_tf_window(sampler)
        self.assert_window(sampler())

    def test_windows_per_image(self):
        with self.assertRaisesRegexp(AssertionError, ''):
            sampler = ImageWindowDataset(reader=get_3d_reader(), batch_size=2,
                                         windows_per_image=2)
            self.assert_window(sampler())

    def test_epoch(self):
        reader = get_3d_reader()
        batch_size = 3
        sampler = ImageWindowDataset(reader=reader, batch_size=batch_size,
                                     epoch=1)
        with self.test_session() as sess:
            sampler.run_threads(sess)
            iters = 0
            try:
                for _ in range(400):
                    window = sess.run(sampler.pop_batch_op())
                    iters = iters + 1
            except tf.errors.OutOfRangeError:
                pass
            # batch size 2, 40 images in total
            self.assertEqual(np.ceil(reader.num_subjects/np.float(batch_size)),
                             iters)

if __name__ == "__main__":
    tf.test.main()
