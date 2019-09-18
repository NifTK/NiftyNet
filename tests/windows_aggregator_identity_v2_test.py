# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import nibabel as nib
import tensorflow as tf
import pandas as pd
import numpy as np
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.engine.windows_aggregator_identity import WindowAsImageAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.utilities.util_common import ParserNamespace
from tests.niftynet_testcase import NiftyNetTestCase

NEW_ORDER = (0, 1, 2, 4, 3)
NEW_ORDER_2D = (0, 1, 3, 2)
MULTI_MOD_DATA = {
    'T1': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1sampler.csv'),
        path_to_search='testing_data',
        filename_contains=('_o_T1_time', '23'),
        filename_not_contains=('Parcellation',),
        interp_order=0,
        pixdim=(2.4, 5.0, 2.0),
        axcodes='LAS',
        spatial_window_size=(23, 32, 15),
        loader=None
    ),
    'FLAIR': ParserNamespace(
        csv_file=os.path.join('testing_data', 'FLAIRsampler.csv'),
        path_to_search='testing_data',
        filename_contains=('FLAIR_', '23'),
        filename_not_contains=('Parcellation',),
        interp_order=0,
        pixdim=(2.4, 5.0, 2.0),
        axcodes='LAS',
        spatial_window_size=(23, 32, 15),
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
        interp_order=0,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(72, 83, 1),
        loader=None
    ),
}
MOD_2D_TASK = ParserNamespace(image=('ultrasound',))

MOD_LABEL_DATA = {
    'parcellation': ParserNamespace(
        csv_file=os.path.join('testing_data', 'Parcelsampler2d.csv'),
        path_to_search='testing_data',
        filename_contains=('23_NeuroMorph_Parcellation',),
        filename_not_contains=('FLAIR',),
        interp_order=0,
        pixdim=None,
        axcodes=None,
        spatial_window_size=(150, 140, 100),
        loader=None
    ),
}
MOD_LABEl_TASK = ParserNamespace(label=('parcellation',))

SINGLE_25D_DATA = {
    'T1': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1sampler.csv'),
        path_to_search='testing_data',
        filename_contains=('_o_T1_time', '106'),
        filename_not_contains=('Parcellation',),
        interp_order=0,
        pixdim=(3.0, 5.0, 5.0),
        axcodes='LAS',
        spatial_window_size=(40, 30, 1),
        loader=None
    ),
}
SINGLE_25D_TASK = ParserNamespace(image=('T1',))

data_partitioner = ImageSetsPartitioner()
multi_mod_list = data_partitioner.initialise(MULTI_MOD_DATA).get_file_list()
mod_2d_list = data_partitioner.initialise(MOD_2D_DATA).get_file_list()
mod_label_list = data_partitioner.initialise(MOD_LABEL_DATA).get_file_list()
single_25d_list = data_partitioner.initialise(SINGLE_25D_DATA).get_file_list()


def get_3d_reader():
    '''
    define the 3d reader
    :return: 3d reader
    '''
    reader = ImageReader(['image'])
    reader.initialise(MULTI_MOD_DATA, MULTI_MOD_TASK, multi_mod_list)
    return reader


def get_2d_reader():
    '''
    define the 2d reader
    :return: 2d reader
    '''
    reader = ImageReader(['image'])
    reader.initialise(MOD_2D_DATA, MOD_2D_TASK, mod_2d_list)
    return reader


def get_label_reader():
    '''
    define the label reader
    :return: label reader
    '''
    reader = ImageReader(['label'])
    reader.initialise(MOD_LABEL_DATA, MOD_LABEl_TASK, mod_label_list)
    label_normaliser = DiscreteLabelNormalisationLayer(
        image_name='label',
        modalities=vars(SINGLE_25D_TASK).get('label'),
        model_filename=os.path.join('testing_data', 'agg_test.txt'))
    reader.add_preprocessing_layers(label_normaliser)
    pad_layer = PadLayer(image_name=('label',), border=(5, 6, 7))
    reader.add_preprocessing_layers([pad_layer])
    return reader


def get_25d_reader():
    '''
    define the 2.5 d reader
    :return:
    '''
    reader = ImageReader(['image'])
    reader.initialise(SINGLE_25D_DATA, SINGLE_25D_TASK, single_25d_list)
    return reader


class IdentityAggregatorTest(NiftyNetTestCase):
    def test_3d_init(self):
        reader = get_3d_reader()
        sampler = ResizeSampler(reader=reader,
                                window_sizes=MULTI_MOD_DATA,
                                batch_size=1,
                                shuffle=False,
                                queue_length=50)
        aggregator = WindowAsImageAggregator(
            image_reader=reader,
            output_path=os.path.join('testing_data', 'aggregated_identity'),
            )
        more_batch = True
        out_shape = []
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                try:
                    out = sess.run(sampler.pop_batch_op())
                    out_shape = out['image'].shape[1:] + (1,)
                except tf.errors.OutOfRangeError:
                    break
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image']}, out['image_location'])
        output_filename = '{}_window_image_niftynet_generated.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        output_file = os.path.join('testing_data',
                                   'aggregated_identity',
                                   output_filename)
        out_shape = [out_shape[i] for i in NEW_ORDER]
        self.assertAllClose(
            nib.load(output_file).shape, out_shape)
        sampler.close_all()

    def test_3d_init_mo(self):
        reader = get_3d_reader()
        sampler = ResizeSampler(reader=reader,
                                window_sizes=MULTI_MOD_DATA,
                                batch_size=1,
                                shuffle=False,
                                queue_length=50)
        aggregator = WindowAsImageAggregator(
            image_reader=reader,
            output_path=os.path.join('testing_data', 'aggregated_identity')
            )
        more_batch = True
        out_shape = []
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                try:
                    out = sess.run(sampler.pop_batch_op())
                    out_shape = out['image'].shape[1:] + (1,)
                except tf.errors.OutOfRangeError:
                    break
                sum_val = np.sum(out['image'])
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'], 'csv_sum': sum_val},
                    out['image_location'])
        output_filename = '{}_window_image_niftynet_generated.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join(
            'testing_data', 'aggregated_identity',
            '{}_csv_sum_niftynet_generated.csv'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated_identity',
                                   output_filename)
        out_shape = [out_shape[i] for i in NEW_ORDER]
        self.assertAllClose(
            nib.load(output_file).shape, out_shape)
        sum_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            sum_pd.shape, [1, 2]
        )
        sampler.close_all()

    def test_3d_init_mo_2im(self):
        reader = get_3d_reader()
        sampler = ResizeSampler(reader=reader,
                                window_sizes=MULTI_MOD_DATA,
                                batch_size=1,
                                shuffle=False,
                                queue_length=50)
        aggregator = WindowAsImageAggregator(
            image_reader=reader,
            output_path=os.path.join('testing_data', 'aggregated_identity'),
            )
        more_batch = True
        out_shape = []
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                try:
                    out = sess.run(sampler.pop_batch_op())
                    out_shape = out['image'].shape[1:] + (1,)
                except tf.errors.OutOfRangeError:
                    break
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'], 'window_im2': out['image']},
                    out['image_location'])
        output_filename = '{}_window_image_niftynet_generated.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        outim2_filename = os.path.join(
            'testing_data', 'aggregated_identity',
            '{}_window_im2_niftynet_generated.nii.gz'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated_identity',
                                   output_filename)
        out_shape = [out_shape[i] for i in NEW_ORDER]
        self.assertAllClose(
            nib.load(output_file).shape, out_shape)
        self.assertAllClose(
            nib.load(outim2_filename).shape, out_shape)
        sampler.close_all()

    def test_3d_init_mo_3out(self):
        reader = get_3d_reader()
        sampler = ResizeSampler(reader=reader,
                                window_sizes=MULTI_MOD_DATA,
                                batch_size=1,
                                shuffle=False,
                                queue_length=50)
        aggregator = WindowAsImageAggregator(
            image_reader=reader,
            output_path=os.path.join('testing_data', 'aggregated_identity')
            )
        more_batch = True
        out_shape = []
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                try:
                    out = sess.run(sampler.pop_batch_op())
                    out_shape = out['image'].shape[1:] + (1,)
                except tf.errors.OutOfRangeError:
                    break
                sum_val = np.sum(out['image'])
                stats_val = [np.sum(out['image']),
                             np.min(out['image']),
                             np.max(out['image'])]
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'], 'csv_sum': sum_val,
                     'csv_stats': stats_val},
                    out['image_location'])
        output_filename = '{}_window_image_niftynet_generated.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join(
            'testing_data', 'aggregated_identity',
            '{}_csv_sum_niftynet_generated.csv'.format(
                sampler.reader.get_subject_id(0)))
        stats_filename = os.path.join(
            'testing_data', 'aggregated_identity',
            '{}_csv_stats_niftynet_generated.csv'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated_identity',
                                   output_filename)
        out_shape = [out_shape[i] for i in NEW_ORDER]
        self.assertAllClose(
            nib.load(output_file).shape, out_shape)
        sum_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            sum_pd.shape, [1, 2]
        )
        stats_pd = pd.read_csv(stats_filename)
        self.assertAllClose(
            stats_pd.shape, [1, 4]
        )
        sampler.close_all()

    def test_init_3d_mo_bidimcsv(self):
        reader = get_3d_reader()
        sampler = ResizeSampler(reader=reader,
                                window_sizes=MULTI_MOD_DATA,
                                batch_size=1,
                                shuffle=False,
                                queue_length=50)
        aggregator = WindowAsImageAggregator(
            image_reader=reader,
            output_path=os.path.join('testing_data', 'aggregated_identity'),
            )
        more_batch = True
        out_shape = []
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                try:
                    out = sess.run(sampler.pop_batch_op())
                    out_shape = out['image'].shape[1:] + (1,)
                except tf.errors.OutOfRangeError:
                    break
                min_val = np.sum((np.asarray(out['image']).flatten()))
                stats_val = [np.min(out['image']), np.max(out['image']), np.sum(
                    out['image'])]
                stats_val = np.expand_dims(stats_val, 0)
                stats_val = np.concatenate([stats_val, stats_val], axis=0)
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'],
                     'csv_sum': min_val,
                     'csv_stats2d': stats_val},
                    out['image_location'])
        output_filename = '{}_window_image_niftynet_generated.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join(
            'testing_data', 'aggregated_identity',
            '{}_csv_sum_niftynet_generated.csv'.format(
                sampler.reader.get_subject_id(0)))
        stats_filename = os.path.join(
            'testing_data', 'aggregated_identity',
            '{}_csv_stats2d_niftynet_generated.csv'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated_identity',
                                   output_filename)
        out_shape = [out_shape[i] for i in NEW_ORDER]
        self.assertAllClose(
            nib.load(output_file).shape, out_shape)
        min_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            min_pd.shape, [1, 2]
        )
        stats_pd = pd.read_csv(stats_filename)
        self.assertAllClose(
            stats_pd.shape, [1, 7]
        )
        sampler.close_all()

    def test_2d_init(self):
        reader = get_2d_reader()
        sampler = ResizeSampler(reader=reader,
                                window_sizes=MOD_2D_DATA,
                                batch_size=1,
                                shuffle=False,
                                queue_length=50)
        aggregator = WindowAsImageAggregator(
            image_reader=reader,
            output_path=os.path.join('testing_data', 'aggregated_identity'),
            )
        more_batch = True
        out_shape = []
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                try:
                    out = sess.run(sampler.pop_batch_op())
                    out_shape = out['image'].shape[1:] + (1,)
                except tf.errors.OutOfRangeError:
                    break
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image']}, out['image_location'])
        output_filename = '{}_window_image_niftynet_generated.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        output_file = os.path.join('testing_data',
                                   'aggregated_identity',
                                   output_filename)
        out_shape = [out_shape[i] for i in NEW_ORDER_2D] + [1,]
        self.assertAllClose(
            nib.load(output_file).shape, out_shape[:2])
        sampler.close_all()

    def test_init_2d_mo(self):
        reader = get_2d_reader()
        sampler = ResizeSampler(reader=reader,
                                window_sizes=MOD_2D_DATA,
                                batch_size=1,
                                shuffle=False,
                                queue_length=50)
        aggregator = WindowAsImageAggregator(
            image_reader=reader,
            output_path=os.path.join('testing_data', 'aggregated_identity'),
            )
        more_batch = True
        out_shape = []
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                try:
                    out = sess.run(sampler.pop_batch_op())
                    out_shape = out['image'].shape[1:] + (1,)
                except tf.errors.OutOfRangeError:
                    break
                min_val = np.sum((np.asarray(out['image']).flatten()))
                stats_val = [np.min(out), np.max(out), np.sum(out)]
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'], 'csv_sum': min_val},
                    out['image_location'])
        output_filename = '{}_window_image_niftynet_generated.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join(
            'testing_data', 'aggregated_identity',
            '{}_csv_sum_niftynet_generated.csv'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated_identity',
                                   output_filename)
        out_shape = [out_shape[i] for i in NEW_ORDER_2D] + [1,]

        self.assertAllClose(
            nib.load(output_file).shape, out_shape[:2])
        min_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            min_pd.shape, [1, 2]
        )
        sampler.close_all()

    def test_init_2d_mo_3out(self):
        reader = get_2d_reader()
        sampler = ResizeSampler(reader=reader,
                                window_sizes=MOD_2D_DATA,
                                batch_size=1,
                                shuffle=False,
                                queue_length=50)
        aggregator = WindowAsImageAggregator(
            image_reader=reader,

            output_path=os.path.join('testing_data', 'aggregated_identity'),
            )
        more_batch = True
        out_shape = []
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                try:
                    out = sess.run(sampler.pop_batch_op())
                    out_shape = out['image'].shape[1:] + (1,)
                except tf.errors.OutOfRangeError:
                    break
                min_val = np.sum((np.asarray(out['image']).flatten()))
                stats_val = [np.min(out['image']), np.max(out['image']), np.sum(
                    out['image'])]
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'],
                     'csv_sum': min_val,
                     'csv_stats': stats_val},
                    out['image_location'])
        output_filename = '{}_window_image_niftynet_generated.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join('testing_data', 'aggregated_identity',
                                    '{}_csv_sum_niftynet_generated.csv'.format(
                                        sampler.reader.get_subject_id(0)))
        stats_filename = os.path.join('testing_data', 'aggregated_identity',
                                      '{}_csv_stats_niftynet_generated.csv'.format(
                                          sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated_identity',
                                   output_filename)
        out_shape = [out_shape[i] for i in NEW_ORDER_2D] + [1,]
        self.assertAllClose(
            nib.load(output_file).shape, out_shape[:2])
        min_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            min_pd.shape, [1, 2]
        )
        stats_pd = pd.read_csv(stats_filename)
        self.assertAllClose(
            stats_pd.shape, [1, 4]
        )
        sampler.close_all()

    def test_init_2d_mo_bidimcsv(self):
        reader = get_2d_reader()
        sampler = ResizeSampler(reader=reader,
                                window_sizes=MOD_2D_DATA,
                                batch_size=1,
                                shuffle=False,
                                queue_length=50)
        aggregator = WindowAsImageAggregator(
            image_reader=reader,

            output_path=os.path.join('testing_data', 'aggregated_identity'),
            )
        more_batch = True
        out_shape = []
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                try:
                    out = sess.run(sampler.pop_batch_op())
                    out_shape = out['image'].shape[1:] + (1,)
                except tf.errors.OutOfRangeError:
                    break
                min_val = np.sum((np.asarray(out['image']).flatten()))
                stats_val = [np.min(out['image']), np.max(out['image']), np.sum(
                    out['image'])]
                stats_val = np.expand_dims(stats_val, 0)
                stats_val = np.concatenate([stats_val, stats_val], axis=0)
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'],
                     'csv_sum': min_val,
                     'csv_stats2d': stats_val},
                    out['image_location'])
        output_filename = '{}_window_image_niftynet_generated.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join('testing_data', 'aggregated_identity',
                                    '{}_csv_sum_niftynet_generated.csv'.format(
                                        sampler.reader.get_subject_id(0)))
        stats_filename = os.path.join('testing_data', 'aggregated_identity',
                                      '{}_csv_stats2d_niftynet_generated.csv'.format(
                                          sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated_identity',
                                   output_filename)

        out_shape = [out_shape[i] for i in NEW_ORDER_2D] + [1,]
        self.assertAllClose(
            nib.load(output_file).shape, out_shape[:2])
        min_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            min_pd.shape, [1, 2]
        )
        stats_pd = pd.read_csv(stats_filename)
        self.assertAllClose(
            stats_pd.shape, [1, 7]
        )
        sampler.close_all()

    def test_25d_init(self):
        reader = get_25d_reader()
        sampler = ResizeSampler(reader=reader,
                                window_sizes=SINGLE_25D_DATA,
                                batch_size=1,
                                shuffle=False,
                                queue_length=50)
        aggregator = WindowAsImageAggregator(
            image_reader=reader,

            output_path=os.path.join('testing_data', 'aggregated_identity'),
            )
        more_batch = True
        out_shape = []
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                try:
                    out = sess.run(sampler.pop_batch_op())
                    out_shape = out['image'].shape[1:] + (1,)
                except tf.errors.OutOfRangeError:
                    break
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image']}, out['image_location'])
        output_filename = '{}_window_image_niftynet_generated.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        output_file = os.path.join('testing_data',
                                   'aggregated_identity',
                                   output_filename)
        out_shape = [out_shape[i] for i in NEW_ORDER_2D] + [1,]
        self.assertAllClose(
            nib.load(output_file).shape, out_shape[:2])
        sampler.close_all()


if __name__ == "__main__":
    tf.test.main()
