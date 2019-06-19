# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.engine.sampler_grid_v2 import \
        GridSampler, _enumerate_step_points, grid_spatial_coordinates
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
        spatial_window_size=(8, 10, 2),
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
        spatial_window_size=(8, 10, 2),
        loader=None
    )
}
MULTI_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'))

MOD_2D_DATA = {
    'ultrasound': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1sampler2d.csv'),
        path_to_search='testing_data',
        filename_contains=('2d_',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(10, 7, 1),
        loader=None
    ),
}
MOD_2D_TASK = ParserNamespace(image=('ultrasound',))

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
DYNAMIC_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'))

data_partitioner = ImageSetsPartitioner()
multi_mod_list = data_partitioner.initialise(MULTI_MOD_DATA).get_file_list()
mod_2d_list = data_partitioner.initialise(MOD_2D_DATA).get_file_list()
dynamic_list = data_partitioner.initialise(DYNAMIC_MOD_DATA).get_file_list()


def get_3d_reader():
    reader = ImageReader(['image'])
    reader.initialise(MULTI_MOD_DATA, MULTI_MOD_TASK, multi_mod_list)
    return reader


def get_2d_reader():
    reader = ImageReader(['image'])
    reader.initialise(MOD_2D_DATA, MOD_2D_TASK, mod_2d_list)
    return reader


def get_dynamic_window_reader():
    reader = ImageReader(['image'])
    reader.initialise(DYNAMIC_MOD_DATA, DYNAMIC_MOD_TASK, dynamic_list)
    return reader


class GridSamplerTest(NiftyNetTestCase):
    def test_3d_initialising(self):
        sampler = GridSampler(reader=get_3d_reader(),
                              window_sizes=MULTI_MOD_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(0, 0, 0),
                              queue_length=10)
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (10, 8, 10, 2, 2))
        sampler.close_all()

    def test_25d_initialising(self):
        sampler = GridSampler(reader=get_3d_reader(),
                              window_sizes=MULTI_MOD_DATA,
                              batch_size=10,
                              spatial_window_size=(1, 20, 15),
                              window_border=(0, 0, 0),
                              queue_length=10)
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (10, 20, 15, 2))
        sampler.close_all()

    def test_2d_initialising(self):
        sampler = GridSampler(reader=get_2d_reader(),
                              window_sizes=MOD_2D_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(0, 0, 0),
                              queue_length=10)
        with self.cached_session() as sess:
            sampler.set_num_threads(1)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (10, 10, 7, 1))
        sampler.close_all()

    def test_dynamic_window_initialising(self):
        sampler = GridSampler(reader=get_dynamic_window_reader(),
                              window_sizes=DYNAMIC_MOD_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(0, 0, 0),
                              queue_length=10)
        with self.cached_session() as sess:
            sampler.set_num_threads(1)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (10, 8, 2, 256, 2))
        sampler.close_all()

    def test_name_mismatch(self):
        with self.assertRaisesRegexp(ValueError, ""):
            sampler = GridSampler(reader=get_dynamic_window_reader(),
                                  window_sizes=MOD_2D_DATA,
                                  batch_size=10,
                                  spatial_window_size=None,
                                  window_border=(0, 0, 0),
                                  queue_length=10)
        with self.assertRaisesRegexp(ValueError, ""):
            sampler = GridSampler(reader=get_3d_reader(),
                                  window_sizes=MOD_2D_DATA,
                                  batch_size=10,
                                  spatial_window_size=None,
                                  window_border=(0, 0, 0),
                                  queue_length=10)


class CoordinatesTest(NiftyNetTestCase):
    def test_coordinates(self):
        coords = grid_spatial_coordinates(
            subject_id=1,
            img_sizes={'image': (64, 64, 64, 1, 2),
                       'label': (42, 42, 42, 1, 1)},
            win_sizes={'image': (63, 63, 40),
                       'label': (42, 41, 33)},
            border_size=(2, 3, 4))
        # first dim corresponds to subject id
        expected_image = np.array(
            [[1, 0, 0, 0, 63, 63, 40],
             [1, 0, 0, 24, 63, 63, 64],
             [1, 0, 0, 12, 63, 63, 52],
             [1, 1, 0, 0, 64, 63, 40],
             [1, 1, 0, 24, 64, 63, 64],
             [1, 1, 0, 12, 64, 63, 52],
             [1, 0, 1, 0, 63, 64, 40],
             [1, 0, 1, 24, 63, 64, 64],
             [1, 0, 1, 12, 63, 64, 52],
             [1, 1, 1, 0, 64, 64, 40],
             [1, 1, 1, 24, 64, 64, 64],
             [1, 1, 1, 12, 64, 64, 52]], dtype=np.int32)
        self.assertAllClose(coords['image'], expected_image)
        expected_label = np.array(
            [[1, 0, 0, 0, 42, 41, 33],
             [1, 0, 0, 9, 42, 41, 42],
             [1, 0, 0, 4, 42, 41, 37],
             [1, 0, 1, 0, 42, 42, 33],
             [1, 0, 1, 9, 42, 42, 42],
             [1, 0, 1, 4, 42, 42, 37]], dtype=np.int32)
        self.assertAllClose(coords['label'], expected_label)
        pass

    def test_2d_coordinates(self):
        coords = grid_spatial_coordinates(
            subject_id=1,
            img_sizes={'image': (64, 64, 1, 1, 2),
                       'label': (42, 42, 1, 1, 1)},
            win_sizes={'image': (63, 63, 1),
                       'label': (30, 32, 1)},
            border_size=(2, 3, 4))
        # first dim corresponds to subject id
        expected_image = np.array(
            [[1, 0, 0, 0, 63, 63, 1],
             [1, 1, 0, 0, 64, 63, 1],
             [1, 0, 1, 0, 63, 64, 1],
             [1, 1, 1, 0, 64, 64, 1]], dtype=np.int32)
        self.assertAllClose(coords['image'], expected_image)
        expected_label = np.array(
            [[1, 0, 0, 0, 30, 32, 1],
             [1, 12, 0, 0, 42, 32, 1],
             [1, 6, 0, 0, 36, 32, 1],
             [1, 0, 10, 0, 30, 42, 1],
             [1, 12, 10, 0, 42, 42, 1],
             [1, 6, 10, 0, 36, 42, 1],
             [1, 0, 5, 0, 30, 37, 1],
             [1, 12, 5, 0, 42, 37, 1],
             [1, 6, 5, 0, 36, 37, 1]], dtype=np.int32)
        self.assertAllClose(coords['label'], expected_label)
        pass

    def test_nopadding_coordinates(self):
        coords = grid_spatial_coordinates(
            subject_id=1,
            img_sizes={'image': (64, 64, 64, 1, 2),
                       'label': (64, 64, 42, 1, 1)},
            win_sizes={'image': (63, 63, 40),
                       'label': (50, 62, 40)},
            border_size=(-1, -1, -1))

        coords_1 = grid_spatial_coordinates(
            subject_id=1,
            img_sizes={'image': (64, 64, 64, 1, 2),
                       'label': (64, 64, 42, 1, 1)},
            win_sizes={'image': (63, 63, 40),
                       'label': (50, 62, 40)},
            border_size=(0, 0, 0))
        self.assertAllClose(coords['image'], coords_1['image'])
        self.assertAllClose(coords['label'], coords_1['label'])
        expected_image = np.array(
            [[1, 0, 0, 0, 63, 63, 40],
             [1, 0, 0, 24, 63, 63, 64],
             [1, 0, 0, 12, 63, 63, 52],
             [1, 1, 0, 0, 64, 63, 40],
             [1, 1, 0, 24, 64, 63, 64],
             [1, 1, 0, 12, 64, 63, 52],
             [1, 0, 1, 0, 63, 64, 40],
             [1, 0, 1, 24, 63, 64, 64],
             [1, 0, 1, 12, 63, 64, 52],
             [1, 1, 1, 0, 64, 64, 40],
             [1, 1, 1, 24, 64, 64, 64],
             [1, 1, 1, 12, 64, 64, 52]], dtype=np.int32)
        self.assertAllClose(coords['image'], expected_image)
        expected_label = np.array(
            [[1, 0, 0, 0, 50, 62, 40],
             [1, 0, 0, 2, 50, 62, 42],
             [1, 0, 0, 1, 50, 62, 41],
             [1, 14, 0, 0, 64, 62, 40],
             [1, 14, 0, 2, 64, 62, 42],
             [1, 14, 0, 1, 64, 62, 41],
             [1, 7, 0, 0, 57, 62, 40],
             [1, 7, 0, 2, 57, 62, 42],
             [1, 7, 0, 1, 57, 62, 41],
             [1, 0, 2, 0, 50, 64, 40],
             [1, 0, 2, 2, 50, 64, 42],
             [1, 0, 2, 1, 50, 64, 41],
             [1, 14, 2, 0, 64, 64, 40],
             [1, 14, 2, 2, 64, 64, 42],
             [1, 14, 2, 1, 64, 64, 41],
             [1, 7, 2, 0, 57, 64, 40],
             [1, 7, 2, 2, 57, 64, 42],
             [1, 7, 2, 1, 57, 64, 41],
             [1, 0, 1, 0, 50, 63, 40],
             [1, 0, 1, 2, 50, 63, 42],
             [1, 0, 1, 1, 50, 63, 41],
             [1, 14, 1, 0, 64, 63, 40],
             [1, 14, 1, 2, 64, 63, 42],
             [1, 14, 1, 1, 64, 63, 41],
             [1, 7, 1, 0, 57, 63, 40],
             [1, 7, 1, 2, 57, 63, 42],
             [1, 7, 1, 1, 57, 63, 41]], dtype=np.int32)
        self.assertAllClose(coords['label'], expected_label)
        with self.assertRaisesRegexp(AssertionError, ""):
            coords_1 = grid_spatial_coordinates(
                subject_id=1,
                img_sizes={'image': (64, 64, 64, 1, 2),
                           'label': (42, 42, 42, 1, 1)},
                win_sizes={'image': (63, 63, 40),
                           'label': (80, 80, 33)},
                border_size=(0, 0, 0))


class StepPointsTest(NiftyNetTestCase):
    def test_steps(self):
        loc = _enumerate_step_points(0, 10, 4, 1)
        self.assertAllClose(loc, [0, 1, 2, 3, 4, 5, 6])

        loc = _enumerate_step_points(0, 10, 4, 2)
        self.assertAllClose(loc, [0, 2, 4, 6])

        loc = _enumerate_step_points(0, 0, 4, 2)
        self.assertAllClose(loc, [0])

        loc = _enumerate_step_points(0, 0, 4, 2)
        self.assertAllClose(loc, [0])

        loc = _enumerate_step_points(0, 0, 0, -1)
        self.assertAllClose(loc, [0])

        loc = _enumerate_step_points(0, 10, 8, 8)
        self.assertAllClose(loc, [0, 2, 1])

        with self.assertRaisesRegexp(ValueError, ""):
            loc = _enumerate_step_points('foo', 0, 0, 10)


if __name__ == "__main__":
    tf.test.main()
