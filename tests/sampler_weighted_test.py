# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.engine.sampler_weighted import WeightedSampler
from niftynet.engine.sampler_weighted import weighted_spatial_coordinates
from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
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
        spatial_window_size=(7, 10, 2)
    ),
    'FLAIR': ParserNamespace(
        csv_file=os.path.join('testing_data', 'FLAIRsampler.csv'),
        path_to_search='testing_data',
        filename_contains=('FLAIR_',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(7, 10, 2)
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
        spatial_window_size=(10, 9, 1)
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


class WeightedSamplerTest(tf.test.TestCase):
    def test_3d_init(self):
        sampler = WeightedSampler(reader=get_3d_reader(),
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
        sampler = WeightedSampler(reader=get_2d_reader(),
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
        sampler = WeightedSampler(reader=get_dynamic_window_reader(),
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
            sampler = WeightedSampler(reader=get_3d_reader(),
                                      data_param=MOD_2D_DATA,
                                      batch_size=2,
                                      windows_per_image=10,
                                      queue_length=10)

    def test_close_early(self):
        sampler = WeightedSampler(reader=get_2d_reader(),
                                  data_param=MOD_2D_DATA,
                                  batch_size=2,
                                  windows_per_image=10,
                                  queue_length=10)
        sampler.close_all()


class RandomCoordinatesTest(tf.test.TestCase):
    def test_coordinates(self):
        coords = weighted_spatial_coordinates(
            subject_id=1,
            data={'sampler': np.random.rand(41, 42, 42, 1, 1)},
            img_sizes={'image': (42, 42, 42, 1, 2),
                       'label': (42, 42, 42, 1, 1)},
            win_sizes={'image': (23, 23, 40),
                       'label': (40, 32, 33)},
            n_samples=10)
        self.assertEquals(np.all(coords['image'][:0] == 1), True)
        self.assertEquals(coords['image'].shape, (10, 7))
        self.assertEquals(coords['label'].shape, (10, 7))
        self.assertAllClose(
            (coords['image'][:, 4] + coords['image'][:, 1]),
            (coords['label'][:, 4] + coords['label'][:, 1]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 5] + coords['image'][:, 2]),
            (coords['label'][:, 5] + coords['label'][:, 2]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 6] + coords['image'][:, 3]),
            (coords['label'][:, 6] + coords['label'][:, 3]), atol=1.0)

    def test_25D_coordinates(self):
        coords = weighted_spatial_coordinates(
            subject_id=1,
            data={'sampler': np.random.rand(42, 42, 42, 1, 1)},
            img_sizes={'image': (42, 42, 42, 1, 1),
                       'label': (42, 42, 42, 1, 1)},
            win_sizes={'image': (23, 23, 1),
                       'label': (40, 32, 1)},
            n_samples=10)
        self.assertEquals(np.all(coords['image'][:0] == 1), True)
        self.assertEquals(coords['image'].shape, (10, 7))
        self.assertEquals(coords['label'].shape, (10, 7))
        self.assertAllClose(
            (coords['image'][:, 4] + coords['image'][:, 1]),
            (coords['label'][:, 4] + coords['label'][:, 1]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 5] + coords['image'][:, 2]),
            (coords['label'][:, 5] + coords['label'][:, 2]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 6] + coords['image'][:, 3]),
            (coords['label'][:, 6] + coords['label'][:, 3]), atol=1.0)

    def test_2D_coordinates(self):
        coords = weighted_spatial_coordinates(
            subject_id=1,
            data={'sampler': np.random.rand(42, 42, 42, 1, 1)},
            img_sizes={'image': (42, 42, 1, 1, 1),
                       'label': (42, 42, 1, 1, 1)},
            win_sizes={'image': (23, 23, 1),
                       'label': (40, 32, 1)},
            n_samples=10)
        self.assertEquals(np.all(coords['image'][:0] == 1), True)
        self.assertEquals(coords['image'].shape, (10, 7))
        self.assertEquals(coords['label'].shape, (10, 7))
        self.assertAllClose(
            (coords['image'][:, 4] + coords['image'][:, 1]),
            (coords['label'][:, 4] + coords['label'][:, 1]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 5] + coords['image'][:, 2]),
            (coords['label'][:, 5] + coords['label'][:, 2]), atol=1.0)
        self.assertAllClose(
            (coords['image'][:, 6] + coords['image'][:, 3]),
            (coords['label'][:, 6] + coords['label'][:, 3]), atol=1.0)

    def test_ill_coordinates(self):
        with self.assertRaisesRegexp(IndexError, ""):
            coords = weighted_spatial_coordinates(
                subject_id=1,
                data={'sampler': np.random.rand(42, 42, 42)},
                img_sizes={'image': (42, 42, 1, 1, 1),
                           'label': (42, 42, 1, 1, 1)},
                win_sizes={'image': (23, 23),
                           'label': (40, 32)},
                n_samples=10)

        with self.assertRaisesRegexp(TypeError, ""):
            coords = weighted_spatial_coordinates(
                subject_id=1,
                data={'sampler': np.random.rand(42, 42, 42, 1, 1)},
                img_sizes={'image': (42, 42, 1, 1, 1),
                           'label': (42, 42, 1, 1, 1)},
                win_sizes={'image': (23, 23, 1),
                           'label': (40, 32, 1)},
                n_samples='test')

        with self.assertRaisesRegexp(AssertionError, ""):
            coords = weighted_spatial_coordinates(
                subject_id=1,
                data={'sampler': np.random.rand(42, 42, 42, 1, 1)},
                img_sizes={'label': (42, 1, 1, 1)},
                win_sizes={'image': (23, 23, 1)},
                n_samples=0)

        with self.assertRaisesRegexp(RuntimeError, ""):
            coords = weighted_spatial_coordinates(
                subject_id=1,
                data={},
                img_sizes={'image': (42, 42, 1, 1, 1),
                           'label': (42, 42, 1, 1, 1)},
                win_sizes={'image': (23, 23, 1),
                           'label': (40, 32, 1)},
                n_samples=10)


if __name__ == "__main__":
    tf.test.main()
