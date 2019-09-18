# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os

import nibabel as nib
import tensorflow as tf
import numpy as np
import pandas as pd
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.utilities.util_common import ParserNamespace
from tests.niftynet_testcase import NiftyNetTestCase

MULTI_MOD_DATA = {
    'T1': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1sampler.csv'),
        path_to_search='testing_data',
        filename_contains=('_o_T1_time', '23'),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=(2.4, 5.0, 2.0),
        axcodes='LSA',
        spatial_window_size=(23, 32, 15),
        loader=None
    ),
    'FLAIR': ParserNamespace(
        csv_file=os.path.join('testing_data', 'FLAIRsampler.csv'),
        path_to_search='testing_data',
        filename_contains=('FLAIR_', '23'),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=(2.4, 5.0, 2.0),
        axcodes='LSA',
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
        interp_order=3,
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
        interp_order=3,
        pixdim=(3.0, 5.0, 5.0),
        axcodes='ASL',
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
    reader = ImageReader(['image'])
    reader.initialise(MULTI_MOD_DATA, MULTI_MOD_TASK, multi_mod_list)
    return reader


def get_2d_reader():
    reader = ImageReader(['image'])
    reader.initialise(MOD_2D_DATA, MOD_2D_TASK, mod_2d_list)
    return reader


def get_label_reader():
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


def get_nonnormalising_label_reader():
    reader = ImageReader(['label'])
    reader.initialise(MOD_LABEL_DATA, MOD_LABEl_TASK, mod_label_list)
    return reader


def get_25d_reader():
    reader = ImageReader(['image'])
    reader.initialise(SINGLE_25D_DATA, SINGLE_25D_TASK, single_25d_list)
    return reader


class GridSamplesAggregatorTest(NiftyNetTestCase):
    def test_3d_init(self):
        reader = get_3d_reader()
        sampler = GridSampler(reader=reader,
                              window_sizes=MULTI_MOD_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(3, 4, 5),
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='image',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=(3, 4, 5),
            interp_order=0)
        more_batch = True

        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())
                more_batch = aggregator.decode_batch(
                    {'window_image':out['image']}, out['image_location'])
        output_filename = 'window_image_{}_niftynet_out.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        output_file = os.path.join('testing_data',
                                   'aggregated',
                                   output_filename)
        self.assertAllClose(
            nib.load(output_file).shape, (256, 168, 256, 1, 2))
        sampler.close_all()

    def test_2d_init(self):
        reader = get_2d_reader()
        sampler = GridSampler(reader=reader,
                              window_sizes=MOD_2D_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(3, 4, 5),
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='image',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=(3, 4, 5),
            interp_order=0)
        more_batch = True
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())
                more_batch = aggregator.decode_batch(
                    {'window_image':out['image']}, out['image_location'])
        output_filename = 'window_image_{}_niftynet_out.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        output_file = os.path.join('testing_data',
                                   'aggregated',
                                   output_filename)
        self.assertAllClose(
            nib.load(output_file).shape, [128, 128])
        sampler.close_all()

    def test_25d_init(self):
        reader = get_25d_reader()
        sampler = GridSampler(reader=reader,
                              window_sizes=SINGLE_25D_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(3, 4, 5),
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='image',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=(3, 4, 5),
            interp_order=0)
        more_batch = True
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())
                more_batch = aggregator.decode_batch(
                    {'window_image':out['image']}, out['image_location'])
        output_filename = 'window_image_{}_niftynet_out.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        output_file = os.path.join('testing_data',
                                   'aggregated',
                                   output_filename)
        print(output_file)
        self.assertAllClose(
            nib.load(output_file).shape, [256, 168, 256],
            rtol=1e-03, atol=1e-03)
        sampler.close_all()

    def test_3d_init_mo(self):
        reader = get_3d_reader()
        sampler = GridSampler(reader=reader,
                              window_sizes=MULTI_MOD_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(3, 4, 5),
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='image',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=(3, 4, 5),
            interp_order=0)
        more_batch = True

        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())
                out_flatten = np.reshape(np.asarray(out['image']), [10, -1])
                min_val = np.sum(np.reshape(np.asarray(out['image']),
                                            [10, -1]), 1)
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'], 'csv_sum': min_val},
                    out['image_location'])
        output_filename = 'window_image_{}_niftynet_out.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join(
            'testing_data', 'aggregated',
            'csv_sum_{}_niftynet_out.csv'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated',
                                   output_filename)

        self.assertAllClose(
            nib.load(output_file).shape, (256, 168, 256, 1, 2))
        min_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            min_pd.shape, [420, 9]
        )
        sampler.close_all()

    def test_3d_init_mo_2im(self):
        reader = get_3d_reader()
        sampler = GridSampler(reader=reader,
                              window_sizes=MULTI_MOD_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(3, 4, 5),
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='image',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=(3, 4, 5),
            interp_order=0)
        more_batch = True

        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())

                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'], 'window_im2': out['image']},
                    out['image_location'])
        output_filename = 'window_image_{}_niftynet_out.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        outim2_filename = os.path.join(
            'testing_data', 'aggregated',
            'window_im2_{}_niftynet_out.nii.gz'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated',
                                   output_filename)
        self.assertAllClose(
            nib.load(output_file).shape, (256, 168, 256, 1, 2))
        self.assertAllClose(
            nib.load(outim2_filename).shape, (256, 168, 256, 1, 2))
        sampler.close_all()

    def test_init_3d_mo_3out(self):
        reader = get_3d_reader()
        sampler = GridSampler(reader=reader,
                              window_sizes=MULTI_MOD_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(3, 4, 5),
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='image',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=(3, 4, 5),
            interp_order=0)
        more_batch = True

        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())
                print(out['image'].shape)
                out_flatten = np.reshape(np.asarray(out['image']), [10, -1])
                min_val = np.sum(np.reshape(
                    np.asarray(out['image']), [10, -1]), 1)
                stats_val = np.concatenate(
                    [np.min(out_flatten, 1, keepdims=True),
                     np.max(out_flatten, 1, keepdims=True),
                     np.sum(out_flatten, 1, keepdims=True)], 1)
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'], 'csv_sum': min_val,
                     'csv_stats': stats_val},
                    out['image_location'])
        output_filename = 'window_image_{}_niftynet_out.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join(
            'testing_data', 'aggregated',
            'csv_sum_{}_niftynet_out.csv'.format(
                sampler.reader.get_subject_id(0)))
        stats_filename = os.path.join(
            'testing_data', 'aggregated',
            'csv_stats_{}_niftynet_out.csv'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated',
                                   output_filename)

        self.assertAllClose(
            nib.load(output_file).shape, (256, 168, 256, 1, 2))
        min_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            min_pd.shape, [420, 9]
        )
        stats_pd = pd.read_csv(stats_filename)
        self.assertAllClose(
            stats_pd.shape, [420, 11]
        )
        sampler.close_all()

    def test_init_3d_mo_bidimcsv(self):
        reader = get_3d_reader()
        sampler = GridSampler(reader=reader,
                              window_sizes=MULTI_MOD_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(3, 4, 5),
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='image',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=(3, 4, 5),
            interp_order=0)
        more_batch = True

        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())
                out_flatten = np.reshape(np.asarray(out['image']), [10, -1])
                min_val = np.sum(np.reshape(
                    np.asarray(out['image']), [10, -1]), 1)
                stats_val = np.concatenate(
                    [np.min(out_flatten, 1, keepdims=True),
                     np.max(out_flatten, 1, keepdims=True),
                     np.sum(out_flatten, 1, keepdims=True)], 1)
                stats_val = np.expand_dims(stats_val, 1)
                stats_val = np.concatenate([stats_val, stats_val], axis=1)
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'],
                     'csv_sum': min_val,
                     'csv_stats_2d': stats_val},
                    out['image_location'])
        output_filename = 'window_image_{}_niftynet_out.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join(
            'testing_data', 'aggregated',
            'csv_sum_{}_niftynet_out.csv'.format(
                sampler.reader.get_subject_id(0)))
        stats_filename = os.path.join(
            'testing_data', 'aggregated',
            'csv_stats_2d_{}_niftynet_out.csv'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated',
                                   output_filename)

        self.assertAllClose(
            nib.load(output_file).shape, (256, 168, 256, 1, 2))
        min_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            min_pd.shape, [420, 9]
        )
        stats_pd = pd.read_csv(stats_filename)
        self.assertAllClose(
            stats_pd.shape, [420, 14]
        )
        sampler.close_all()

    def test_init_2d_mo(self):
        reader = get_2d_reader()
        sampler = GridSampler(reader=reader,
                              window_sizes=MOD_2D_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(3, 4, 5),
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='image',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=(3, 4, 5),
            interp_order=0)
        more_batch = True

        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())
                out_flatten = np.reshape(np.asarray(out['image']), [10, -1])
                min_val = np.sum(np.reshape(
                    np.asarray(out['image']), [10, -1]), 1)
                stats_val = np.concatenate(
                    [np.min(out_flatten, 1, keepdims=True),
                     np.max(out_flatten, 1, keepdims=True),
                     np.sum(out_flatten, 1, keepdims=True)], 1)
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'], 'csv_sum': min_val},
                    out['image_location'])
        output_filename = 'window_image_{}_niftynet_out.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join(
            'testing_data', 'aggregated',
            'csv_sum_{}_niftynet_out.csv'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated',
                                   output_filename)

        self.assertAllClose(
            nib.load(output_file).shape, (128, 128))
        min_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            min_pd.shape, [10, 9]
        )
        sampler.close_all()

    def test_init_2d_mo_3out(self):
        reader = get_2d_reader()
        sampler = GridSampler(reader=reader,
                              window_sizes=MOD_2D_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(3, 4, 5),
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='image',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=(3, 4, 5),
            interp_order=0)
        more_batch = True

        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())
                out_flatten = np.reshape(np.asarray(out['image']), [10, -1])
                min_val = np.sum(np.reshape(
                    np.asarray(out['image']), [10, -1]), 1)
                stats_val = np.concatenate(
                    [np.min(out_flatten, 1, keepdims=True),
                     np.max(out_flatten, 1, keepdims=True),
                     np.sum(out_flatten, 1, keepdims=True)], 1)
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'],
                     'csv_sum': min_val,
                     'csv_stats': stats_val},
                    out['image_location'])
        output_filename = 'window_image_{}_niftynet_out.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join(
            'testing_data', 'aggregated',
            'csv_sum_{}_niftynet_out.csv'.format(
                sampler.reader.get_subject_id(0)))
        stats_filename = os.path.join(
            'testing_data', 'aggregated',
            'csv_stats_{}_niftynet_out.csv'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated',
                                   output_filename)

        self.assertAllClose(
            nib.load(output_file).shape, (128, 128))
        min_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            min_pd.shape, [10, 9]
        )
        stats_pd = pd.read_csv(stats_filename)
        self.assertAllClose(
            stats_pd.shape, [10, 11]
        )
        sampler.close_all()

    def test_init_2d_mo_bidimcsv(self):
        reader = get_2d_reader()
        sampler = GridSampler(reader=reader,
                              window_sizes=MOD_2D_DATA,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(3, 4, 5),
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='image',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=(3, 4, 5),
            interp_order=0)
        more_batch = True

        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())
                out_flatten = np.reshape(np.asarray(out['image']), [10, -1])
                min_val = np.sum(np.reshape(
                    np.asarray(out['image']), [10, -1]), 1)
                stats_val = np.concatenate(
                    [np.min(out_flatten, 1, keepdims=True),
                     np.max(out_flatten, 1, keepdims=True),
                     np.sum(out_flatten, 1, keepdims=True)], 1)
                stats_val = np.expand_dims(stats_val, 1)
                stats_val = np.concatenate([stats_val, stats_val], axis=1)
                more_batch = aggregator.decode_batch(
                    {'window_image': out['image'],
                     'csv_sum': min_val,
                     'csv_stats_2d': stats_val},
                    out['image_location'])
        output_filename = 'window_image_{}_niftynet_out.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        sum_filename = os.path.join(
            'testing_data', 'aggregated',
            'csv_sum_{}_niftynet_out.csv'.format(
                sampler.reader.get_subject_id(0)))
        stats_filename = os.path.join(
            'testing_data', 'aggregated',
            'csv_stats_2d_{}_niftynet_out.csv'.format(
                sampler.reader.get_subject_id(0)))
        output_file = os.path.join('testing_data',
                                   'aggregated',
                                   output_filename)

        self.assertAllClose(
            nib.load(output_file).shape, (128, 128))
        min_pd = pd.read_csv(sum_filename)
        self.assertAllClose(
            min_pd.shape, [10, 9]
        )
        stats_pd = pd.read_csv(stats_filename)
        self.assertAllClose(
            stats_pd.shape, [10, 14]
        )
        sampler.close_all()

    def test_inverse_mapping(self):
        reader = get_label_reader()
        data_param = MOD_LABEL_DATA
        sampler = GridSampler(reader=reader,
                              window_sizes=data_param,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=(3, 4, 5),
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='label',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=(3, 4, 5),
            interp_order=0)
        more_batch = True
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())
                more_batch = aggregator.decode_batch(
                    {'window_label': out['label']}, out['label_location'])
        output_filename = 'window_label_{}_niftynet_out.nii.gz'.format(
            sampler.reader.get_subject_id(0))
        output_file = os.path.join(
            'testing_data', 'aggregated', output_filename)
        self.assertAllClose(
            nib.load(output_file).shape, [256, 168, 256])
        sampler.close_all()
        output_data = nib.load(output_file).get_data()
        expected_data = nib.load(
            'testing_data/T1_1023_NeuroMorph_Parcellation.nii.gz').get_data()
        self.assertAllClose(output_data, expected_data)

    def test_filling(self):
        reader = get_nonnormalising_label_reader()
        test_constant = 0.5731
        postfix = '_niftynet_out_background'
        test_border = (10, 7, 8)
        data_param = MOD_LABEL_DATA
        sampler = GridSampler(reader=reader,
                              window_sizes=data_param,
                              batch_size=10,
                              spatial_window_size=None,
                              window_border=test_border,
                              queue_length=50)
        aggregator = GridSamplesAggregator(
            image_reader=reader,
            name='label',
            output_path=os.path.join('testing_data', 'aggregated'),
            window_border=test_border,
            interp_order=0,
            postfix=postfix,
            fill_constant=test_constant)
        more_batch = True
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            while more_batch:
                out = sess.run(sampler.pop_batch_op())
                more_batch = aggregator.decode_batch(
                    {'window_label': out['label']}, out['label_location'])
        output_filename = 'window_label_{}_{}.nii.gz'.format(
            sampler.reader.get_subject_id(0), postfix)
        output_file = os.path.join(
            'testing_data', 'aggregated', output_filename)
        output_data = nib.load(output_file).get_data()
        output_shape = output_data.shape
        for i in range(3):
            def _test_background(idcs):
                extract = output_data[idcs]
                self.assertTrue((extract == test_constant).sum()
                                == extract.size)

            extract_idcs = [slice(None)]*3

            extract_idcs[i] = slice(0, test_border[i])
            _test_background(tuple(extract_idcs))

            extract_idcs[i] = slice(output_shape[i] - test_border[i],
                                    output_shape[i])
            _test_background(tuple(extract_idcs))


if __name__ == "__main__":
    tf.test.main()
