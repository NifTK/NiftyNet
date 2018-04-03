# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import unittest

import numpy as np
import tensorflow as tf

from niftynet.engine.sampler_balanced import BalancedSampler
from niftynet.engine.sampler_balanced import balanced_spatial_coordinates
from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.utilities.util_common import ParserNamespace
from niftynet.engine.image_window import N_SPATIAL

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


class BalancedSamplerTest(tf.test.TestCase):
    def test_3d_init(self):
        sampler = BalancedSampler(reader=get_3d_reader(),
                                  data_param=MULTI_MOD_DATA,
                                  batch_size=2,
                                  windows_per_image=10,
                                  queue_length=10)
        with self.test_session() as sess:
            coordinator = tf.train.Coordinator()
            sampler.run_threads(sess, coordinator, num_threads=2)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (2, 7, 10, 2, 2))
        sampler.close_all()

    def test_2d_init(self):
        sampler = BalancedSampler(reader=get_2d_reader(),
                                  data_param=MOD_2D_DATA,
                                  batch_size=2,
                                  windows_per_image=10,
                                  queue_length=10)
        with self.test_session() as sess:
            coordinator = tf.train.Coordinator()
            sampler.run_threads(sess, coordinator, num_threads=2)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape, (2, 10, 9, 1))
        sampler.close_all()

    def test_dynamic_init(self):
        sampler = BalancedSampler(reader=get_dynamic_window_reader(),
                                  data_param=DYNAMIC_MOD_DATA,
                                  batch_size=2,
                                  windows_per_image=10,
                                  queue_length=10)
        with self.test_session() as sess:
            coordinator = tf.train.Coordinator()
            sampler.run_threads(sess, coordinator, num_threads=2)
            with self.assertRaisesRegexp(tf.errors.OutOfRangeError, ""):
                out = sess.run(sampler.pop_batch_op())

    def test_ill_init(self):
        with self.assertRaisesRegexp(KeyError, ""):
            sampler = BalancedSampler(reader=get_3d_reader(),
                                      data_param=MOD_2D_DATA,
                                      batch_size=2,
                                      windows_per_image=10,
                                      queue_length=10)

    def test_close_early(self):
        sampler = BalancedSampler(reader=get_2d_reader(),
                                  data_param=MOD_2D_DATA,
                                  batch_size=2,
                                  windows_per_image=10,
                                  queue_length=10)
        sampler.close_all()


class BalancedCoordinatesTest(tf.test.TestCase):
    def assertCoordinatesAreValid(self, coords, cropped_map):
        for coord in coords:
            for i in range(len(coord.shape)):
                self.assertTrue(coord[i] >= 0)
                self.assertTrue(coord[i] < cropped_map.shape[i])

    def test_3d_coordinates(self):
        cropped_map=np.zeros((256, 512, 128))
        coords = balanced_spatial_coordinates(
            cropped_map=cropped_map,
            n_samples=32)

        self.assertAllEqual(coords.shape, (32, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, cropped_map)

    def test_2d_coordinates(self):
        cropped_map=np.zeros((256, 512, 1))
        coords = balanced_spatial_coordinates(
            cropped_map=cropped_map,
            n_samples=64)

        self.assertAllEqual(coords.shape, (64, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, cropped_map)

    def test_repeated_coordinates(self):
        cropped_map=np.zeros((1, 1, 1))
        coords = balanced_spatial_coordinates(
            cropped_map=cropped_map,
            n_samples=10)

        self.assertAllEqual(coords.shape, (10, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, cropped_map)

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
        cropped_map = np.zeros((100, 100, 100))
        cropped_map[0, 0, 0] = 1
        cropped_map[0, 0, 1] = 2

        # Accumulate the number of times each class is sampled
        accum = np.zeros((num_classes))
        for _ in range(number_of_repetitions):
            coords = balanced_spatial_coordinates(
                cropped_map=cropped_map,
                n_samples=samples_per_repetition)

            # Be sure to sample the correct number
            self.assertAllEqual(coords.shape, (samples_per_repetition, N_SPATIAL))

            # Convert to np.ndarry indexable
            for coord in coords.astype(int):
                x,y,z = coord
                label = int(cropped_map[x, y, z])
                accum[label] = accum[label] + 1

        # Each class should be within 2 decimal places of 1.0/num_classes
        accum = np.divide(accum, accum.sum())
        self.assertAllClose(accum, np.ones((num_classes))*1.0/num_classes, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tf.test.main()
