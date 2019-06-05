# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import N_SPATIAL
from niftynet.engine.sampler_weighted_v2 import \
    WeightedSampler, weighted_spatial_coordinates
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


class WeightedSamplerTest(NiftyNetTestCase):
    def test_3d_init(self):
        sampler = WeightedSampler(reader=get_3d_reader(),
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
        sampler = WeightedSampler(reader=get_2d_reader(),
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
        sampler = WeightedSampler(reader=get_dynamic_window_reader(),
                                  window_sizes=DYNAMIC_MOD_DATA,
                                  batch_size=2,
                                  windows_per_image=10,
                                  queue_length=10)
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape[1:], (8, 2, 256, 2))

    def test_ill_init(self):
        with self.assertRaisesRegexp(ValueError, ""):
            sampler = WeightedSampler(reader=get_3d_reader(),
                                      window_sizes=MOD_2D_DATA,
                                      batch_size=2,
                                      windows_per_image=10,
                                      queue_length=10)

    def test_close_early(self):
        sampler = WeightedSampler(reader=get_2d_reader(),
                                  window_sizes=MOD_2D_DATA,
                                  batch_size=2,
                                  windows_per_image=10,
                                  queue_length=10)
        sampler.close_all()


class WeightedCoordinatesTest(NiftyNetTestCase):
    def assertCoordinatesAreValid(self, coords, sampling_map):
        for coord in coords:
            for i in range(len(coord.shape)):
                self.assertTrue(coord[i] >= 0)
                self.assertTrue(coord[i] < sampling_map.shape[i])

    def test_3d_coordinates(self):
        img_size = (32, 16, 17, 1, 1)
        win_size = (10, 16, 15)
        sampling_map = np.zeros(img_size)

        coords = weighted_spatial_coordinates(
            32, img_size, win_size, sampling_map)
        self.assertAllEqual(coords.shape, (32, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, sampling_map)

        # testing high weight location (10, 8, 7, 0, 0)
        sampling_map[10, 8, 7, 0, 0] = 1.0
        coords = weighted_spatial_coordinates(
            32, img_size, win_size, sampling_map)
        self.assertAllEqual(coords.shape, (32, N_SPATIAL))
        self.assertTrue(np.all(coords == [10, 8, 7]))

    def test_2d_coordinates(self):
        img_size = (32, 17, 1, 1, 1)
        win_size = (31, 3, 1)
        sampling_map = np.zeros(img_size)
        coords = weighted_spatial_coordinates(
            64, img_size, win_size, sampling_map)

        self.assertAllEqual(coords.shape, (64, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, sampling_map)

        # testing high weight location (15, 1, 1, 0, 0)
        sampling_map[15, 1, 0, 0, 0] = 1.0
        coords = weighted_spatial_coordinates(
            64, img_size, win_size, sampling_map)
        self.assertAllEqual(coords.shape, (64, N_SPATIAL))
        self.assertTrue(np.all(coords == [15, 1, 0]))

    def test_1d_coordinates(self):
        img_size = (32, 1, 1, 1, 1)
        win_size = (15, 1, 1)
        sampling_map = np.zeros(img_size)
        coords = weighted_spatial_coordinates(
            10, img_size, win_size, sampling_map)
        self.assertAllEqual(coords.shape, (10, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, sampling_map)

        sampling_map[20, 0, 0] = 0.1
        coords = weighted_spatial_coordinates(
            10, img_size, win_size, sampling_map)
        self.assertAllEqual(coords.shape, (10, N_SPATIAL))
        self.assertTrue(np.all(coords == [20, 0, 0]))

        sampling_map[9, 0, 0] = 0.1
        coords = weighted_spatial_coordinates(
            10, img_size, win_size, sampling_map)
        self.assertAllEqual(coords.shape, (10, N_SPATIAL))
        self.assertTrue(np.all((coords == [20, 0, 0]) | (coords == [9, 0, 0])))

if __name__ == "__main__":
    tf.test.main()
