# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.contrib.segmentation_selective_sampler.sampler_selective import \
    SelectiveSampler, Constraint
from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.utilities.util_common import ParserNamespace
from tests.niftynet_testcase import NiftyNetTestCase


### utility function for testing purposes
def check_constraint(data, constraint):
    unique, count = np.unique(np.round(data), return_counts=True)
    list_labels = []
    data = np.round(data)
    if constraint.list_labels is not None:
        list_labels = constraint.list_labels
        for label in constraint.list_labels:
            if label not in unique:
                print('Label %d is not there' % label)
                return False
    num_labels_add = 0
    if constraint.num_labels > 0:
        num_labels_add = constraint.num_labels - len(list_labels)
        if num_labels_add <= 0:
            num_labels_add = 0
        if len(unique) < constraint.num_labels:
            print('Missing labels')
            return False
    to_add = num_labels_add
    if constraint.min_ratio > 0:
        num_min = constraint.min_number_from_ratio(data.shape)
        print('unique in test is ', unique)
        for value, c in zip(unique, count):
            if value in list_labels:
                if c < num_min:
                    print('Not enough in label %d', value)
                    return False
            else:
                if c > num_min:
                    to_add -= 1
        if to_add > 0:
            print('to add initial is ', num_labels_add)
            print('Not enough in additional labels')
            return False
    return True


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
    ),
    'Label': ParserNamespace(
        csv_file=os.path.join('testing_data', 'lesion.csv'),
        path_to_search='testing_data',
        filename_contains=('LesionFin_',),
        filename_not_constains=('FLAIR_',),
        interp_order=1,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(7, 10, 2),
        loader=None
    )
}
LABEL_TASK = {
    'Lesion': ParserNamespace(
        csv_file=os.path.join('testing_data', 'lesion.csv'),
        path_to_search='testing_data',
        filename_contains=('LesionFin_'),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(7, 10, 2),
        loader=None
    )
}
MULTI_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'), label=('Label',))

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
    ),
    'Label': ParserNamespace(
        csv_file=os.path.join('testing_data', 'labels.csv'),
        path_to_search='testing_data',
        filename_contains=('T1_', '_NeuroMorph_Parcellation',),
        filename_not_constains=('FLAIR_',),
        interp_order=1,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(8, 2),
        loader=None
    )
}

# LABEL_TASK = {
#     'Parcellation': ParserNamespace(
#         csv_file=os.path.join('testing_data', 'labels.csv'),
#         path_to_search='testing_data',
#         filename_contains=('Parcellation',),
#         filename_not_constains=('FLAIR_',),
#         interp_order=1,
#         pixdim=None,
#         axcodes=None,
#         spatial_window_size=(8,2)
#     )
# }

DYNAMIC_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'), label=('Label',))

data_partitioner = ImageSetsPartitioner()


def get_3d_reader():
    multi_mod_list = data_partitioner.initialise(MULTI_MOD_DATA).get_file_list()
    print(MULTI_MOD_DATA, MULTI_MOD_TASK)
    reader = ImageReader(['image', 'label'])
    reader.initialise(MULTI_MOD_DATA, MULTI_MOD_TASK, multi_mod_list)
    return reader


def get_2d_reader():
    mod_2d_list = data_partitioner.initialise(MOD_2D_DATA).get_file_list()
    reader = ImageReader(['image'])
    reader.initialise(MOD_2D_DATA, MOD_2D_TASK, mod_2d_list)
    return reader


def get_dynamic_window_reader():
    dynamic_list = data_partitioner.initialise(DYNAMIC_MOD_DATA).get_file_list()
    reader = ImageReader(['image', 'label'])
    reader.initialise(DYNAMIC_MOD_DATA, DYNAMIC_MOD_TASK, dynamic_list)
    return reader


class SelectiveSamplerTest(NiftyNetTestCase):
    def test_3d_init(self):
        constraint_built = Constraint(compulsory_labels=[1],
                                      min_ratio=0.000001,
                                      min_num_labels=2)
        sampler = SelectiveSampler(reader=get_3d_reader(),
                                   data_param=MULTI_MOD_DATA,
                                   batch_size=2,
                                   constraint=constraint_built,
                                   windows_per_image=2,
                                   queue_length=10)
        with self.cached_session() as sess:
            sampler.run_threads(sess, num_threads=1)
            out = sess.run(sampler.pop_batch_op())
            self.assertTrue(check_constraint(out['label'], constraint_built))
            self.assertAllClose(out['image'].shape, (2, 7, 10, 2, 2))
            self.assertAllClose(out['label'].shape, (2, 7, 10, 2, 1))
            print("Test should finish here")
        sampler.close_all()

    # def test_2d_init(self):
    #     sampler = UniformSampler(reader=get_2d_reader(),
    #                              data_param=MOD_2D_DATA,
    #                              batch_size=2,
    #                              windows_per_image=10,
    #                              queue_length=10)
    #     with self.cached_session() as sess:
    #         sampler.run_threads(sess, num_threads=2)
    #         out = sess.run(sampler.pop_batch_op())
    #         self.assertAllClose(out['image'].shape, (2, 10, 9, 1))
    #     sampler.close_all()

    # def test_dynamic_init(self):
    #     sampler = SelectiveSampler(reader=get_dynamic_window_reader(),
    #                                data_param=DYNAMIC_MOD_DATA,
    #                                batch_size=2,
    #                                constraint=Constraint(
    #                                    compulsory_labels=[1],
    #                                    min_ratio=0.000001,
    #                                    min_num_labels=2),
    #                                windows_per_image=2,
    #                                queue_length=2)
    #     with self.cached_session() as sess:
    #         sampler.run_threads(sess, num_threads=2)
    #         out = sess.run(sampler.pop_batch_op())
    #         test = np.zeros_like(out['label'])
    #         test[out['label'] == 1] = 1
    #         print('Number label 52 is ', np.sum(test))
    #         print(out['image_location'], out['label'].shape)
    #         self.assertAllClose(out['image'].shape, (1, 8, 2, 256, 2))
    #         self.assertAllClose(out['label'].shape, (1, 8, 2, 256, 1))
    #     sampler.close_all()

    # def test_ill_init(self):
    #    with self.assertRaisesRegexp(KeyError, ""):
    #        sampler = SelectiveSampler(reader=get_3d_reader(),
    #                                   data_param=MOD_2D_DATA,
    #                                   batch_size=2,
    #                                   windows_per_image=10,
    #                                   queue_length=10)

    # def test_close_early(self):
    #    sampler = SelectiveSampler(reader=get_3d_reader(),
    #                               data_param=DYNAMIC_MOD_DATA,
    #                               batch_size=2,
    #                               windows_per_image=10,
    #                               queue_length=10)
    #    sampler.close_all()


# class RandomCoordinatesTest(NiftyNetTestCase):
#     def test_coordinates(self):
#         coords = rand_choice_coordinates(
#             subject_id=1,
#             img_sizes={'image': (42, 42, 42, 1, 2),
#                        'label': (42, 42, 42, 1, 1)},
#             win_sizes={'image': (23, 23, 40),
#                        'label': (40, 32, 33)},
#             candidates=np.round(np.random.random((256,256,168,1,1))),
#             n_samples=10,
#             mean_counts_size=None)
#         self.assertEquals(np.all(coords['image'][:0] == 1), True)
#         self.assertEquals(coords['image'].shape, (10, 7))
#         self.assertEquals(coords['label'].shape, (10, 7))
#         self.assertAllClose(
#             (coords['image'][:, 4] + coords['image'][:, 1]),
#             (coords['label'][:, 4] + coords['label'][:, 1]), atol=1.0)
#         self.assertAllClose(
#             (coords['image'][:, 5] + coords['image'][:, 2]),
#             (coords['label'][:, 5] + coords['label'][:, 2]), atol=1.0)
#         self.assertAllClose(
#             (coords['image'][:, 6] + coords['image'][:, 3]),
#             (coords['label'][:, 6] + coords['label'][:, 3]), atol=1.0)
#
#
#     def test_ill_coordinates(self):
#         with self.assertRaisesRegexp(IndexError, ""):
#             coords = rand_choice_coordinates(
#                 subject_id=1,
#                 img_sizes={'image': (42, 42, 1, 1, 1),
#                            'label': (42, 42, 1, 1, 1)},
#                 win_sizes={'image': (23, 23),
#                            'label': (40, 32)},
#                 candidates=np.round(np.random.random((256,256,168,1,1))),
#                 n_samples=10,
#                 mean_counts_size=None)
#
#         with self.assertRaisesRegexp(TypeError, ""):
#             coords = rand_choice_coordinates(
#                 subject_id=1,
#                 img_sizes={'image': (42, 42, 1, 1, 1),
#                            'label': (42, 42, 1, 1, 1)},
#                 win_sizes={'image': (23, 23, 1),
#                            'label': (40, 32, 1)},
#                 candidates=np.round(np.random.random((256, 256, 168, 1, 1))),
#                 n_samples='test',
#                 mean_counts_size=None)
#
#         with self.assertRaisesRegexp(AssertionError, ""):
#             coords = rand_choice_coordinates(
#                 subject_id=1,
#                 img_sizes={'label': (42, 1, 1, 1)},
#                 win_sizes={'image': (23, 23, 1)},
#                 candidates=np.round(np.random.random((256, 256, 168, 1, 1))),
#                 n_samples=0,
#                 mean_counts_size=None)


if __name__ == "__main__":
    tf.test.main()
