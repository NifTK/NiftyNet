# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import tensorflow as tf

from niftynet.io.image_sets_partitioner import COLUMN_UNIQ_ID
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.engine.signal import TRAIN, VALID, INFER
from niftynet.utilities.util_common import ParserNamespace
from tests.niftynet_testcase import NiftyNetTestCase

test_sections = {
    'T1': ParserNamespace(
        csv_file=os.path.join('testing_data', 'test_reader.csv'),
        path_to_search='testing_data',
        filename_contains=('_o_T1_time',),
        filename_not_contain=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        loader=None),

    'Flair': ParserNamespace(
        csv_file=os.path.join('testing_data', 'test_Flairreader.csv'),
        path_to_search='testing_data',
        filename_contains=('FLAIR_',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        loader=None)}

partition_output = os.path.join('testing_data', 'partition.csv')


class ImageSetsPartitionerTest(NiftyNetTestCase):
    def test_no_partition_file(self):
        if os.path.isfile(partition_output):
            os.remove(partition_output)

        data_param = test_sections
        test_partitioner = ImageSetsPartitioner()
        test_partitioner.initialise(
            data_param,
            new_partition=False,
            data_split_file=partition_output)
        self.assertEqual(
            test_partitioner.get_file_list()[COLUMN_UNIQ_ID].count(), 4)
        with self.assertRaisesRegexp(ValueError, ''):
            test_partitioner.get_file_list(TRAIN)
        with self.assertRaisesRegexp(ValueError, ''):
            test_partitioner.get_file_list(VALID)
        with self.assertRaisesRegexp(ValueError, ''):
            test_partitioner.get_file_list(INFER)


class ImageSetsPartitionerNewPartition(NiftyNetTestCase):
    def test_new_partition(self):
        data_param = test_sections
        test_partitioner = ImageSetsPartitioner()
        with self.assertRaisesRegexp(TypeError, ''):
            test_partitioner.initialise(
                data_param,
                new_partition=True,
                data_split_file=partition_output)
        test_partitioner.initialise(
            data_param,
            new_partition=True,
            ratios=(2.0, 2.0),
            data_split_file=partition_output)
        self.assertEqual(
            test_partitioner.get_file_list()[COLUMN_UNIQ_ID].count(), 4)
        self.assertEqual(
            test_partitioner.get_file_list(TRAIN), None)
        self.assertEqual(
            test_partitioner.get_file_list(VALID)[COLUMN_UNIQ_ID].count(), 4)
        self.assertEqual(
            test_partitioner.get_file_list(INFER), None)
        self.assertEqual(
            test_partitioner.get_file_list(
                VALID, 'T1', 'Flair')[COLUMN_UNIQ_ID].count(), 4)
        self.assertEqual(
            test_partitioner.get_file_list(
                VALID, 'Flair')[COLUMN_UNIQ_ID].count(), 4)
        with self.assertRaisesRegexp(ValueError, ''):
            test_partitioner.get_file_list(VALID, 'foo')
        with self.assertRaisesRegexp(ValueError, ''):
            test_partitioner.get_file_list('T1')

        self.assertFalse(test_partitioner.has_training)
        self.assertFalse(test_partitioner.has_inference)
        self.assertTrue(test_partitioner.has_validation)


class ImageSetsPartitionerIllPartition(NiftyNetTestCase):
    def test_incompatible_partition_file(self):
        self._reset_partition_file()
        # adding invalid line
        with open(partition_output, 'a') as partition_file:
            partition_file.write('foo, bar')
        test_partitioner = ImageSetsPartitioner()
        with self.assertRaisesRegexp(ValueError, ""):
            test_partitioner.initialise(
                test_sections,
                new_partition=False,
                data_split_file=partition_output)

    def test_replicated_ids(self):
        self._reset_partition_file()
        with open(partition_output, 'a') as partition_file:
            partition_file.write('1065,Training\n')
            partition_file.write('1065,Validation')
        test_partitioner = ImageSetsPartitioner()
        test_partitioner.initialise(
            test_sections,
            new_partition=False,
            data_split_file=partition_output)
        self.assertEqual(
            test_partitioner.get_file_list()[COLUMN_UNIQ_ID].count(), 4)
        self.assertEqual(
            test_partitioner.get_file_list(TRAIN)[COLUMN_UNIQ_ID].count(), 3)
        self.assertEqual(
            test_partitioner.get_file_list(VALID)[COLUMN_UNIQ_ID].count(), 2)
        self.assertEqual(
            test_partitioner.get_file_list(INFER)[COLUMN_UNIQ_ID].count(), 1)

    def test_empty(self):
        self._reset_partition_file()
        with open(partition_output, 'w') as partition_file:
            partition_file.write('')
        test_partitioner = ImageSetsPartitioner()
        with self.assertRaisesRegexp(ValueError, ""):
            test_partitioner.initialise(
                test_sections,
                new_partition=False,
                data_split_file=partition_output)

    def _reset_partition_file(self):
        test_partitioner = ImageSetsPartitioner()
        test_partitioner.initialise(
            test_sections,
            new_partition=True,
            ratios=(0.2, 0.2),
            data_split_file=partition_output)


if __name__ == "__main__":
    tf.test.main()
