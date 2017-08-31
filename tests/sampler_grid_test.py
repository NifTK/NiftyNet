# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.engine.sampler_grid import GridSampler
from niftynet.engine.sampler_grid import _enumerate_step_points
from niftynet.engine.sampler_grid import grid_spatial_coordinates
from niftynet.io.image_reader import ImageReader
from tests.test_util import ParserNamespace

MULTI_MOD_DATA = {
    'T1': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1sampler.csv'),
        path_to_search='testing_data',
        filename_contains=('_o_T1_time',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(8, 10, 2)
    ),
    'FLAIR': ParserNamespace(
        csv_file=os.path.join('testing_data', 'FLAIRsampler.csv'),
        path_to_search='testing_data',
        filename_contains=('FLAIR_',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(8, 10, 2)
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
        spatial_window_size=(10, 7, 1)
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
        spatial_window_size=(8, 2)
    ),
    'FLAIR': ParserNamespace(
        csv_file=os.path.join('testing_data', 'FLAIRsampler.csv'),
        path_to_search='testing_data',
        filename_contains=('FLAIR_',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(8, 2)
    )
}
DYNAMIC_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'))


def get_3d_reader():
    reader = ImageReader(['image'])
    reader.initialise_reader(MULTI_MOD_DATA, MULTI_MOD_TASK)
    return reader


def get_2d_reader():
    reader = ImageReader(['image'])
    reader.initialise_reader(MOD_2D_DATA, MOD_2D_TASK)
    return reader


def get_dynamic_window_reader():
    reader = ImageReader(['image'])
    reader.initialise_reader(DYNAMIC_MOD_DATA, DYNAMIC_MOD_TASK)
    return reader


class GridSamplerTest(tf.test.TestCase):
    def test_3d_initialising(self):
        sampler = GridSampler(reader=get_3d_reader(),
                              data_param=MULTI_MOD_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(0, 0, 0),
                              queue_length=10)
        with self.test_session() as sess:
            coordinator = tf.train.Coordinator()
            sampler.run_threads(sess, coordinator, num_threads=2)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (10, 8, 10, 2, 2))
        sampler.close_all()

    def test_25d_initialising(self):
        sampler = GridSampler(reader=get_3d_reader(),
                              data_param=MULTI_MOD_DATA,
                              batch_size=10,
                              spatial_window_size=(1, 20, 15),
                              window_border=(0, 0, 0),
                              queue_length=10)
        with self.test_session() as sess:
            coordinator = tf.train.Coordinator()
            sampler.run_threads(sess, coordinator, num_threads=2)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (10, 20, 15, 2))
        sampler.close_all()

    def test_2d_initialising(self):
        sampler = GridSampler(reader=get_2d_reader(),
                              data_param=MOD_2D_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(0, 0, 0),
                              queue_length=10)
        with self.test_session() as sess:
            coordinator = tf.train.Coordinator()
            sampler.run_threads(sess, coordinator, num_threads=1)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (10, 10, 7, 1))
        sampler.close_all()

    def test_dynamic_window_initialising(self):
        sampler = GridSampler(reader=get_dynamic_window_reader(),
                              data_param=DYNAMIC_MOD_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(0, 0, 0),
                              queue_length=10)
        with self.test_session() as sess:
            coordinator = tf.train.Coordinator()
            sampler.run_threads(sess, coordinator, num_threads=1)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (1, 8, 2, 256, 2))
        sampler.close_all()

    def test_name_mismatch(self):
        with self.assertRaisesRegexp(KeyError, ""):
            sampler = GridSampler(reader=get_dynamic_window_reader(),
                                  data_param=MOD_2D_DATA,
                                  batch_size=10,
                                  spatial_window_size=None,
                                  window_border=(0, 0, 0),
                                  queue_length=10)
        with self.assertRaisesRegexp(KeyError, ""):
            sampler = GridSampler(reader=get_3d_reader(),
                                  data_param=MOD_2D_DATA,
                                  batch_size=10,
                                  spatial_window_size=None,
                                  window_border=(0, 0, 0),
                                  queue_length=10)


class CoordinatesTest(tf.test.TestCase):
    def test_coordinates(self):
        coords = grid_spatial_coordinates(
            subject_id=1,
            img_sizes={'image': (64, 64, 64, 1, 2),
                       'label': (42, 42, 42, 1, 1)},
            win_sizes={'image': (63, 63, 40),
                       'label': (30, 32, 33)},
            border_size=(2, 3, 4))
        # first dim cooresponds to subject id
        expected_image = np.array(
            [[1, 0, 0, 0, 63, 63, 40],
             [1, 0, 0, 24, 63, 63, 64],
             [1, 1, 0, 0, 64, 63, 40],
             [1, 1, 0, 24, 64, 63, 64],
             [1, 0, 1, 0, 63, 64, 40],
             [1, 0, 1, 24, 63, 64, 64],
             [1, 1, 1, 0, 64, 64, 40],
             [1, 1, 1, 24, 64, 64, 64]], dtype=np.int32)
        self.assertAllClose(coords['image'], expected_image)
        expected_label = np.array(
            [[1, 0, 0, 0, 30, 32, 33],
             [1, 0, 0, 9, 30, 32, 42],
             [1, 12, 0, 0, 42, 32, 33],
             [1, 12, 0, 9, 42, 32, 42],
             [1, 0, 10, 0, 30, 42, 33],
             [1, 0, 10, 9, 30, 42, 42],
             [1, 12, 10, 0, 42, 42, 33],
             [1, 12, 10, 9, 42, 42, 42]], dtype=np.int32)
        self.assertAllClose(coords['label'], expected_label)
        pass

    def test_nopadding_coordinates(self):
        coords = grid_spatial_coordinates(
            subject_id=1,
            img_sizes={'image': (64, 64, 64, 1, 2),
                       'label': (42, 42, 42, 1, 1)},
            win_sizes={'image': (63, 63, 40),
                       'label': (40, 32, 33)},
            border_size=(-1, -1, -1))

        coords_1 = grid_spatial_coordinates(
            subject_id=1,
            img_sizes={'image': (64, 64, 64, 1, 2),
                       'label': (42, 42, 42, 1, 1)},
            win_sizes={'image': (63, 63, 40),
                       'label': (40, 32, 33)},
            border_size=(0, 0, 0))
        self.assertAllClose(coords['image'], coords_1['image'])
        self.assertAllClose(coords['label'], coords_1['label'])
        expected_image = np.array(
            [[1, 0, 0, 0, 63, 63, 40],
             [1, 0, 0, 24, 63, 63, 64],
             [1, 1, 0, 0, 64, 63, 40],
             [1, 1, 0, 24, 64, 63, 64],
             [1, 0, 1, 0, 63, 64, 40],
             [1, 0, 1, 24, 63, 64, 64],
             [1, 1, 1, 0, 64, 64, 40],
             [1, 1, 1, 24, 64, 64, 64]], dtype=np.int32)
        self.assertAllClose(coords['image'], expected_image)
        expected_label = np.array(
            [[1, 0, 0, 0, 40, 32, 33],
             [1, 0, 0, 9, 40, 32, 42],
             [1, 2, 0, 0, 42, 32, 33],
             [1, 2, 0, 9, 42, 32, 42],
             [1, 0, 10, 0, 40, 42, 33],
             [1, 0, 10, 9, 40, 42, 42],
             [1, 2, 10, 0, 42, 42, 33],
             [1, 2, 10, 9, 42, 42, 42]], dtype=np.int32)
        self.assertAllClose(coords['label'], expected_label)
        with self.assertRaisesRegexp(AssertionError, ""):
            coords_1 = grid_spatial_coordinates(
                subject_id=1,
                img_sizes={'image': (64, 64, 64, 1, 2),
                           'label': (42, 42, 42, 1, 1)},
                win_sizes={'image': (63, 63, 40),
                           'label': (80, 80, 33)},
                border_size=(0, 0, 0))


class StepPointsTest(tf.test.TestCase):
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
        self.assertAllClose(loc, [0, 2])

        with self.assertRaisesRegexp(ValueError, ""):
            loc = _enumerate_step_points('foo', 0, 0, 10)


if __name__ == "__main__":
    tf.test.main()
