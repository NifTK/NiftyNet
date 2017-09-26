# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.engine.sampler_selective import SelectiveSampler
from niftynet.engine.sampler_selective import Constraint
from niftynet.engine.sampler_selective import rand_choice_coordinates
from niftynet.engine.sampler_selective import check_constraint
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
    ),
    'L': ParserNamespace(
        csv_file=os.path.join('testing_data', 'lesion.csv'),
        path_to_search='testing_data',
        filename_contains=('LesionFin_',),
        filename_not_constains=('FLAIR_',),
        interp_order=1,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(7, 10, 2)
    )
}
LABEL_TASK = {
    'Lesion': ParserNamespace(
        csv_file=os.path.join('testing_data','lesion.csv'),
        path_to_search='testing_data',
        filename_contains=('LesionFin_'),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(7, 10, 2)
    )
}
MULTI_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'), label=('L'))

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
    ),
    'L': ParserNamespace(
        csv_file=os.path.join('testing_data', 'labels.csv'),
        path_to_search='testing_data',
        filename_contains=('T1_','_NeuroMorph_Parcellation',),
        filename_not_constains=('FLAIR_',),
        interp_order=1,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(8,2)
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
DYNAMIC_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'), label='L')


def get_3d_reader():
    print(MULTI_MOD_DATA, MULTI_MOD_TASK)
    reader = ImageReader(['image', 'label'])
    reader.initialise_reader(MULTI_MOD_DATA, MULTI_MOD_TASK)
    return reader


def get_2d_reader():
    reader = ImageReader(['image'])
    reader.initialise_reader(MOD_2D_DATA, MOD_2D_TASK)
    return reader


def get_dynamic_window_reader():
    reader = ImageReader(['image','label'])
    reader.initialise_reader(DYNAMIC_MOD_DATA, DYNAMIC_MOD_TASK)
    return reader


class SelectiveSamplerTest(tf.test.TestCase):
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
        with self.test_session() as sess:
            coordinator = tf.train.Coordinator()
            sampler.run_threads(sess, coordinator, num_threads=2)
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
    #     with self.test_session() as sess:
    #         coordinator = tf.train.Coordinator()
    #         sampler.run_threads(sess, coordinator, num_threads=2)
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
    #     with self.test_session() as sess:
    #         coordinator = tf.train.Coordinator()
    #         sampler.run_threads(sess, coordinator, num_threads=2)
    #         out = sess.run(sampler.pop_batch_op())
    #         test = np.zeros_like(out['label'])
    #         test[out['label'] == 1] = 1
    #         print('Number label 52 is ', np.sum(test))
    #         print(out['image_location'], out['label'].shape)
    #         self.assertAllClose(out['image'].shape, (1, 8, 2, 256, 2))
    #         self.assertAllClose(out['label'].shape, (1, 8, 2, 256, 1))
    #     sampler.close_all()

    def test_ill_init(self):
        with self.assertRaisesRegexp(KeyError, ""):
            sampler = SelectiveSampler(reader=get_3d_reader(),
                                       data_param=MOD_2D_DATA,
                                       batch_size=2,
                                       windows_per_image=10,
                                       queue_length=10)

    def test_close_early(self):
        sampler = SelectiveSampler(reader=get_3d_reader(),
                                   data_param=DYNAMIC_MOD_DATA,
                                   batch_size=2,
                                   windows_per_image=10,
                                   queue_length=10)
        sampler.close_all()


class RandomCoordinatesTest(tf.test.TestCase):
    def test_coordinates(self):
        coords = rand_choice_coordinates(
            subject_id=1,
            img_sizes={'image': (42, 42, 42, 1, 2),
                       'label': (42, 42, 42, 1, 1)},
            win_sizes={'image': (23, 23, 40),
                       'label': (40, 32, 33)},
            candidates=np.round(np.random.random((256,256,168,1,1))),
            n_samples=10,
            mean_counts_size=None)
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
            rand_choice_coordinates()
            coords = rand_choice_coordinates(
                subject_id=1,
                img_sizes={'image': (42, 42, 1, 1, 1),
                           'label': (42, 42, 1, 1, 1)},
                win_sizes={'image': (23, 23),
                           'label': (40, 32)},
                candidates=np.round(np.random.random((256,256,168,1,1))),
                n_samples=10,
                mean_counts_size=None)

        with self.assertRaisesRegexp(TypeError, ""):
            coords = rand_choice_coordinates(
                subject_id=1,
                img_sizes={'image': (42, 42, 1, 1, 1),
                           'label': (42, 42, 1, 1, 1)},
                win_sizes={'image': (23, 23, 1),
                           'label': (40, 32, 1)},
                candidates=np.round(np.random.random((256, 256, 168, 1, 1))),
                n_samples='test',
                mean_counts_size=None)

        with self.assertRaisesRegexp(AssertionError, ""):
            coords = rand_choice_coordinates(
                subject_id=1,
                img_sizes={'label': (42, 1, 1, 1)},
                win_sizes={'image': (23, 23, 1)},
                candidates=np.round(np.random.random((256, 256, 168, 1, 1))),
                n_samples=0,
                mean_counts_size=None)


if __name__ == "__main__":
    tf.test.main()
