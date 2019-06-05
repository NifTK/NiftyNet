# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import unittest

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import N_SPATIAL
from niftynet.engine.sampler_balanced_v2 import \
    BalancedSampler, balanced_spatial_coordinates
from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.utilities.util_common import ParserNamespace
from tests.niftynet_testcase import NiftyNetTestCase

MULTI_MOD_DATA = {
    'T1': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1sampler.csv'),
        path_to_search='testing_data',
        filename_contains=('_o_T1_time',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(7, 10, 2),
        loader=None
    ),
    'FLAIR': ParserNamespace(
        csv_file=os.path.join('testing_data', 'FLAIRsampler.csv'),
        path_to_search='testing_data',
        filename_contains=('FLAIR_',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(7, 10, 2),
        loader=None
    )
}
MULTI_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'),
                                 sampler=('T1',))

MOD_2D_DATA = {
    'ultrasound': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1sampler2d.csv'),
        path_to_search='testing_data',
        filename_contains=('2d_',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(10, 9, 1),
        loader=None
    ),
}
MOD_2D_TASK = ParserNamespace(image=('ultrasound',),
                              sampler=('ultrasound',))

DYNAMIC_MOD_DATA = {
    'T1': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1sampler.csv'),
        path_to_search='testing_data',
        filename_contains=('_o_T1_time',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(8, 2),
        loader=None
    ),
    'FLAIR': ParserNamespace(
        csv_file=os.path.join('testing_data', 'FLAIRsampler.csv'),
        path_to_search='testing_data',
        filename_contains=('FLAIR_',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(8, 2),
        loader=None
    )
}
DYNAMIC_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'),
                                   sampler=('FLAIR',))

data_partitioner = ImageSetsPartitioner()
multi_mod_list = data_partitioner.initialise(MULTI_MOD_DATA).get_file_list()
mod_2d_list = data_partitioner.initialise(MOD_2D_DATA).get_file_list()
dynamic_list = data_partitioner.initialise(DYNAMIC_MOD_DATA).get_file_list()


def get_3d_reader():
    reader = ImageReader(['image', 'sampler'])
    reader.initialise(MULTI_MOD_DATA, MULTI_MOD_TASK, multi_mod_list)
    return reader


def get_2d_reader():
    reader = ImageReader(['image', 'sampler'])
    reader.initialise(MOD_2D_DATA, MOD_2D_TASK, mod_2d_list)
    return reader


def get_dynamic_window_reader():
    reader = ImageReader(['image', 'sampler'])
    reader.initialise(DYNAMIC_MOD_DATA, DYNAMIC_MOD_TASK, dynamic_list)
    return reader


@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class BalancedSamplerTest(NiftyNetTestCase):
    def test_3d_init(self):
        sampler = BalancedSampler(reader=get_3d_reader(),
                                  window_sizes=MULTI_MOD_DATA,
                                  batch_size=2,
                                  windows_per_image=10,
                                  queue_length=10)
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (2, 7, 10, 2, 2))
        sampler.close_all()

    def test_2d_init(self):
        sampler = BalancedSampler(reader=get_2d_reader(),
                                  window_sizes=MOD_2D_DATA,
                                  batch_size=2,
                                  windows_per_image=10,
                                  queue_length=10)
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (2, 10, 9, 1))
        sampler.close_all()

    def test_dynamic_init(self):
        sampler = BalancedSampler(reader=get_dynamic_window_reader(),
                                  window_sizes=DYNAMIC_MOD_DATA,
                                  batch_size=2,
                                  windows_per_image=10,
                                  queue_length=10)
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            #with self.assertRaisesRegexp(tf.errors.OutOfRangeError, ""):
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape[1:], (8, 2, 256, 2))

    def test_ill_init(self):
        with self.assertRaisesRegexp(ValueError, ""):
            sampler = BalancedSampler(reader=get_3d_reader(),
                                      window_sizes=MOD_2D_DATA,
                                      batch_size=2,
                                      windows_per_image=10,
                                      queue_length=10)

    def test_close_early(self):
        sampler = BalancedSampler(reader=get_2d_reader(),
                                  window_sizes=MOD_2D_DATA,
                                  batch_size=2,
                                  windows_per_image=10,
                                  queue_length=10)
        sampler.close_all()


class BalancedCoordinatesTest(NiftyNetTestCase):
    def assertCoordinatesAreValid(self, coords, sampling_map):
        for coord in coords:
            for i in range(len(coord.shape)):
                self.assertTrue(coord[i] >= 0)
                self.assertTrue(coord[i] < sampling_map.shape[i])

    def test_3d_coordinates(self):
        img_size = (64, 15, 21, 1, 1)
        win_size = (32, 13, 1, 1, 1)
        sampling_map = np.zeros(img_size)
        coords = balanced_spatial_coordinates(
            32, img_size, win_size, sampling_map)

        self.assertAllEqual(coords.shape, (32, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, sampling_map)

        sampling_map[17, 7, 0, 0, 0] = 1.0
        coords = balanced_spatial_coordinates(
            500, img_size, win_size, sampling_map)
        # better test?
        self.assertTrue(np.sum(np.all(coords == [17, 7, 0], axis=1)) >= 200)

    def test_2d_coordinates(self):
        img_size = (23, 42, 1, 1, 1)
        win_size = (22, 10, 1)
        sampling_map = np.zeros(img_size)

        coords = balanced_spatial_coordinates(
            64, img_size, win_size, sampling_map)

        self.assertAllEqual(coords.shape, (64, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, sampling_map)

        sampling_map[11, 8, 0, 0, 0] = 1.0
        coords = balanced_spatial_coordinates(
            500, img_size, win_size, sampling_map)
        # better test?
        self.assertTrue(np.sum(np.all(coords == [11, 8, 0], axis=1)) >= 200)

    def test_1d_coordinates(self):
        img_size = (21, 1, 1, 1, 1)
        win_size = (15, 1, 1)
        sampling_map = np.zeros(img_size)
        coords = balanced_spatial_coordinates(
            10, img_size, win_size, sampling_map)

        self.assertAllEqual(coords.shape, (10, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, sampling_map)

        sampling_map[9, 0, 0, 0, 0] = 1.0
        coords = balanced_spatial_coordinates(
            500, img_size, win_size, sampling_map)
        # better test?
        self.assertTrue(np.sum(np.all(coords == [9, 0, 0], axis=1)) >= 200)

    @unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
    def test_classes_balances(self):
        # Set the random state to prevent false positive
        np.random.seed(0)

        # Setting these too high inflats the run time
        number_of_repetitions = 1000
        samples_per_repetition = 10
        num_classes = 3

        # Create a map with almost all background, one pixel of each
        # other label
        img_size = (50, 25, 10, 1, 1)
        win_size = (8, 7, 2, 1, 1)
        sampling_map = np.zeros(img_size)
        sampling_map[6, 5, 2:, 0, 0] = 1
        sampling_map[11, 13:, 3, 0, 0] = 2

        # Accumulate the number of times each class is sampled
        accum = np.zeros((num_classes))
        for _ in range(number_of_repetitions):
            coords = balanced_spatial_coordinates(
                samples_per_repetition, img_size, win_size, sampling_map)

            # Be sure to sample the correct number
            self.assertAllEqual(
                coords.shape, (samples_per_repetition, N_SPATIAL))

            # Convert to np.ndarry indexable
            for coord in coords.astype(int):
                x, y, z = coord
                label = int(sampling_map[x, y, z])
                accum[label] = accum[label] + 1

        # Each class should be within 2 decimal places of 1.0/num_classes
        accum = np.divide(accum, accum.sum())
        self.assertAllClose(
            accum,
            np.ones((num_classes)) * 1.0 / num_classes, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tf.test.main()
