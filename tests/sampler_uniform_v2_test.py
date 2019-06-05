# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

# from niftynet.engine.sampler_uniform import UniformSampler
from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.engine.sampler_uniform_v2 import rand_spatial_coordinates
from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.utilities.util_common import ParserNamespace
from niftynet.engine.image_window import N_SPATIAL
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
MULTI_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'))

MULTI_WINDOW_DATA = {
    'T1': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1sampler.csv'),
        path_to_search='testing_data',
        filename_contains=('_o_T1_time',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(4, 10, 3),
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
        spatial_window_size=(7, 12, 2),
        loader=None
    )
}
MULTI_WINDOW_TASK = ParserNamespace(image=('T1',), label=('FLAIR',))

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

def get_concentric_window_reader():
    reader = ImageReader(['image', 'label'])
    reader.initialise(MULTI_WINDOW_DATA, MULTI_WINDOW_TASK, multi_mod_list)
    return reader


class UniformSamplerTest(NiftyNetTestCase):
    def test_3d_concentric_init(self):
        sampler = UniformSampler(reader=get_concentric_window_reader(),
                                 window_sizes=MULTI_WINDOW_DATA,
                                 batch_size=2,
                                 windows_per_image=10,
                                 queue_length=10)
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            out = sess.run(sampler.pop_batch_op())
            img_loc = out['image_location']
            seg_loc = out['label_location']
            self.assertTrue(np.all(img_loc[:, 0] == seg_loc[:, 0]))
            self.assertTrue(np.all((img_loc - seg_loc)[:, 1:4] == [1, 1, 0]))
            self.assertTrue(np.all((img_loc - seg_loc)[:, 4:] == [-2, -1, 1]))
            self.assertAllClose(out['image'].shape, (2, 4, 10, 3, 1))
            self.assertAllClose(out['label'].shape, (2, 7, 12, 2, 1))
        sampler.close_all()

    def test_3d_init(self):
        sampler = UniformSampler(reader=get_3d_reader(),
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
        sampler = UniformSampler(reader=get_2d_reader(),
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
        sampler = UniformSampler(reader=get_dynamic_window_reader(),
                                 window_sizes=DYNAMIC_MOD_DATA,
                                 batch_size=2,
                                 windows_per_image=10,
                                 queue_length=10)
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['image'].shape[1:], (8, 2, 256, 2))
        sampler.close_all()

    def test_ill_init(self):
        with self.assertRaisesRegexp(ValueError, ""):
            sampler = UniformSampler(reader=get_3d_reader(),
                                     window_sizes=MOD_2D_DATA,
                                     batch_size=2,
                                     windows_per_image=10,
                                     queue_length=10)

    def test_close_early(self):
        sampler = UniformSampler(reader=get_dynamic_window_reader(),
                                 window_sizes=DYNAMIC_MOD_DATA,
                                 batch_size=2,
                                 windows_per_image=10,
                                 queue_length=10)
        sampler.close_all()


class RandomCoordinatesTest(NiftyNetTestCase):
    def assertCoordinatesAreValid(self, coords, img_size, win_size):
        for coord in coords:
            for i in range(len(coord.shape)):
                self.assertTrue(coord[i] >= int(win_size[i] / 2))
                self.assertTrue(coord[i] <= img_size[i] - int(win_size[i]/2))

    def test_3d_coordinates(self):
        img_size = [8, 9, 10]
        win_size = [7, 9, 4]
        coords = rand_spatial_coordinates(
            32, img_size, win_size, None)
        self.assertAllEqual(coords.shape, (32, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, img_size, win_size)

    def test_2d_coordinates(self):
        cropped_map=np.zeros((256, 512, 1))
        img_size = [8, 9, 1]
        win_size = [8, 8, 1]
        coords = rand_spatial_coordinates(
            64, img_size, win_size, None)
        self.assertAllEqual(coords.shape, (64, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, img_size, win_size)

    def test_1d_coordinates(self):
        cropped_map=np.zeros((1, 1, 1))
        img_size = [4, 1, 1]
        win_size = [2, 1, 1]
        coords = rand_spatial_coordinates(
            20, img_size, win_size, None)
        print(coords)
        self.assertAllEqual(coords.shape, (20, N_SPATIAL))
        self.assertCoordinatesAreValid(coords, img_size, win_size)


if __name__ == "__main__":
    tf.test.main()
