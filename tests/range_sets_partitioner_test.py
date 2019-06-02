# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
from functools import reduce

import tensorflow as tf

from niftynet.engine.signal import TRAIN, VALID, INFER, ALL
from niftynet.io.range_sets_partitioner import RangeSetsPartitioner
from niftynet.utilities.util_common import ParserNamespace
from niftynet.io.base_sets_partitioner import COLUMN_UNIQ_ID

NUM_SUBJECTS = 100

partition_output = os.path.join('testing_data', 'memory_partition.csv')

data_param = {'input': ParserNamespace(pixdim=(),
                                       axcodes=(),
                                       filename_contains='some-string',
                                       interp_order=3),
              'output': ParserNamespace(pixdim=(),
                                        axcodes=(),
                                        filename_contains='another-string',
                                        interp_order=0)}

action_param = ParserNamespace(output_postfix='never-read',
                               num_classes=2,
                               output_interp_order=1,
                               spatial_window_size=(80, 80))

app_param = ParserNamespace(compulsory_labels=(0, 1),
                            image=('input',),
                            label=('output',))

class MemoryImageSetsPartitionerTest(tf.test.TestCase):
    def test_file_split_file_error(self):
        if os.path.isfile(partition_output):
            os.remove(partition_output)

        with open(partition_output, 'w') as fout:
            for sub in range(10):
                fout.write('sub%i,training\n' % sub)

        test_partitioner = RangeSetsPartitioner(
            NUM_SUBJECTS, ('input', 'output'), data_split_file='./notexistfile')
        with self.assertRaisesRegexp(ValueError, ''):
            test_partitioner.initialise(new_partition=False).get_file_list(TRAIN)

    def test_ratio_error(self):
        test_partitioner = RangeSetsPartitioner(
            NUM_SUBJECTS,
            ('input', 'output'),
            data_split_file=None)
        test_partitioner.initialise(
            new_partition=True,
            ratios=(0.0, 0.5))
        val_idcs = test_partitioner.get_file_lists_by(phase=VALID)[0]
        self.assertTrue(val_idcs==None)

    def test_partitioning(self):
        test_partitioner = RangeSetsPartitioner(
            NUM_SUBJECTS,
            ('input', 'output'),
            data_split_file=None)

        for val_frac in (0.1, 0.2, 0.4, 0.49):
            test_partitioner.initialise(
                new_partition=True,
                ratios=(val_frac, 0.5 - val_frac))

            ref_train = int(0.5*NUM_SUBJECTS)
            ref_val = int(val_frac*NUM_SUBJECTS)
            ref_infer = int((0.5 - val_frac)*NUM_SUBJECTS)

            all_idcs = test_partitioner.get_file_lists_by(phase=ALL)[0]
            id_list = all_idcs[COLUMN_UNIQ_ID]
            self.assertTrue(id_list.count() == NUM_SUBJECTS
                            and len(set(id_list)) == NUM_SUBJECTS
                            and min(id_list) == '0'
                            and max(id_list) == '{}'.format(NUM_SUBJECTS - 1))

            train_idcs = test_partitioner.get_file_lists_by(phase=TRAIN)[0]
            train_ids = train_idcs[COLUMN_UNIQ_ID]
            self.assertTrue(abs(len(train_ids) - ref_train) <= 1)
            val_idcs = test_partitioner.get_file_lists_by(phase=VALID)[0]
            val_ids = val_idcs[COLUMN_UNIQ_ID]
            self.assertTrue(abs(len(val_ids) - ref_val) <= 1
                            and not any(idx in val_ids for idx in train_ids))
            infer_idcs = test_partitioner.get_file_lists_by(phase=INFER)[0]
            infer_ids = infer_idcs[COLUMN_UNIQ_ID]
            self.assertTrue(abs(len(infer_ids) - ref_infer) <= 1
                            and not any(idx in infer_ids for idx in train_ids)
                            and not any(idx in infer_ids for idx in val_ids))
            self.assertEqual(len(train_ids) + len(val_ids) + len(infer_ids),
                             NUM_SUBJECTS)


if __name__ == '__main__':
    tf.test.main()
