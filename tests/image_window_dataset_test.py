# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window_dataset import ImageWindowDataset
from niftynet.io.image_reader import ImageReader
from tests.niftynet_testcase import NiftyNetTestCase

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


class ImageWindowDataset_2D_Test(NiftyNetTestCase):
    def assert_window(self, window):
        if not isinstance(window, dict):
            window = next(window)
        self.assertEqual(window['mr'].shape[1:3], (120, 160))
        self.assertEqual(window['mr_location'][0, 1:].tolist(),
                         [0, 0, 0, 120, 160, 1])
        self.assertEqual(window['mr'].dtype, np.float32)
        self.assertEqual(window['mr_location'].dtype, np.int32)

    def assert_tf_window(self, sampler):
        with self.cached_session() as sess:
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

    # # sampler layer_op()'s output shape is not checked
    # def test_wrong_window_size_dict(self):
    #    sampler = ImageWindowDataset(reader=get_2d_reader(),
    #                                 batch_size=2,
    #                                 window_sizes=(3,3,0))
    #    self.assert_tf_window(sampler)

    def test_windows_per_image(self):
        sampler = ImageWindowDataset(reader=get_2d_reader(), batch_size=2,
                                     windows_per_image=2)
        self.assert_window(sampler())

    def test_epoch(self):
        reader = get_2d_reader()
        batch_size = 3
        sampler = ImageWindowDataset(
            reader=reader, batch_size=batch_size, epoch=1)
        with self.cached_session() as sess:
            next_element = sampler.pop_batch_op()
            iters = 0
            try:
                for _ in range(400):
                    window = sess.run(next_element)
                    iters = iters + 1
            except tf.errors.OutOfRangeError:
                pass
            # batch size 3, 40 images in total
            self.assertEqual(
                np.ceil(reader.num_subjects / np.float(batch_size)), iters)


class ImageWindowDataset_3D_Test(NiftyNetTestCase):
    def assert_window(self, window):
        if not isinstance(window, dict):
            window = next(window)
        self.assertEqual(window['mr'].shape[1:4], (256, 168, 256))
        self.assertEqual(window['mr_location'][0, 1:].tolist(),
                         [0, 0, 0, 256, 168, 256])
        self.assertEqual(window['mr'].dtype, np.float32)
        self.assertEqual(window['mr_location'].dtype, np.int32)

    def assert_tf_window(self, sampler):
        with self.cached_session() as sess:
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
        sampler = ImageWindowDataset(reader=get_3d_reader(), batch_size=2,
                                     windows_per_image=2)
        self.assert_window(sampler())

    def test_epoch(self):
        reader = get_3d_reader()
        batch_size = 3
        sampler = ImageWindowDataset(
            reader=reader, batch_size=batch_size, epoch=1)
        with self.cached_session() as sess:
            next_element = sampler.pop_batch_op()
            iters = 0
            try:
                for _ in range(400):
                    window = sess.run(next_element)
                    iters = iters + 1
            except tf.errors.OutOfRangeError:
                pass
            # batch size 3, 4 images in total
            self.assertEqual(
                np.ceil(reader.num_subjects / np.float(batch_size)), iters)


class ImageDatasetParamTest(NiftyNetTestCase):
    def run_dataset(self, n_iters, n_threads, **kwargs):
        sampler = ImageWindowDataset(**kwargs)
        sampler.set_num_threads(n_threads)
        with self.cached_session() as sess:
            true_iters = 0
            next_element = sampler.pop_batch_op()
            windows = []
            try:
                for _ in range(min(n_iters, 100)):
                    windows.append(
                        sess.run(next_element)['mr_location'])
                    true_iters = true_iters + 1
            except (tf.errors.OutOfRangeError, EOFError):
                pass
            assert true_iters <= 100, 'keep the test smaller than 100 iters'
        return true_iters, np.concatenate(windows, 0)

    def test_function(self):
        reader = get_2d_reader()
        #### with default batch padding
        n_iters, windows = self.run_dataset(
            n_iters=2,
            n_threads=4,
            reader=reader,
            batch_size=100,
            smaller_final_batch_mode='pad',
            epoch=4)
        # elements: 4 * 40, batch size 100, resulting 2 batches
        self.assertEqual(n_iters, 2)
        self.assertEqual(windows.shape[0], 200)
        # all subjects evaluated
        uniq, counts = np.unique(windows[:, 0], return_counts=True)
        self.assertEqual(len(uniq), 41)
        self.assertTrue(np.all(counts[1:] == 4))

        #### with drop batch
        n_iters, windows = self.run_dataset(
            n_iters=2,
            n_threads=3,
            reader=reader,
            batch_size=100,
            smaller_final_batch_mode='drop',
            epoch=3)
        # elements: 4 * 40, batch size 100, resulting 1 batches
        self.assertEqual(n_iters, 1)
        self.assertEqual(windows.shape[0], 100)
        # all subjects evaluated, might not get all unique items
        # self.assertEqual(len(np.unique(windows[:, 0])), 40)

        #### with drop batch
        n_iters, windows = self.run_dataset(
            n_iters=2,
            n_threads=4,
            reader=reader,
            batch_size=100,
            queue_length=100,
            smaller_final_batch_mode='dynamic',
            epoch=4)
        # elements: 4 * 40, batch size 100, resulting 2 batches
        self.assertEqual(n_iters, 2)
        self.assertEqual(windows.shape[0], 160)
        # all subjects evaluated
        uniq, counts = np.unique(windows[:, 0], return_counts=True)
        self.assertEqual(len(uniq), 40)
        self.assertTrue(np.all(counts == 4))


if __name__ == "__main__":
    tf.test.main()
