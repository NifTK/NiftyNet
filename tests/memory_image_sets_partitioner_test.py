# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
from functools import reduce

import tensorflow as tf

from niftynet.engine.signal import TRAIN, VALID, INFER, ALL
from niftynet.io.range_sets_partitioner import RangeSetsPartitioner
from niftynet.utilities.util_common import ParserNamespace

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
            NUM_SUBJECTS, ('input', 'output'), data_split_file=partition_output)
        with self.assertRaisesRegexp(ValueError, 'F/S'):
            test_partitioner.initialise(new_partition=False)

    def test_partitioning(self):
        test_partitioner = RangeSetsPartitioner(
            NUM_SUBJECTS,
            ('input', 'output'),
            data_split_file=None)

        for val_frac in (0.0, 0.2, 0.4):
            test_partitioner.initialise(
                new_partition=False,
                ratios=(val_frac, 0.5 - val_frac))

            ref_train = int(0.5*NUM_SUBJECTS)
            ref_val = int(val_frac*NUM_SUBJECTS)
            ref_infer = int((0.5 - val_frac)*NUM_SUBJECTS)

            all_idcs = test_partitioner.get_file_lists_by(phase=ALL)
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
            all_idcs = reduce(lambda lst, idcs: lst + idcs, all_idcs, [])
            self.assertTrue(len(all_idcs) == NUM_SUBJECTS
                            and len(set(all_idcs)) == NUM_SUBJECTS
                            and min(all_idcs) == 0
                            and max(all_idcs) == NUM_SUBJECTS - 1)

            train_idcs = test_partitioner.get_file_lists_by(phase=TRAIN)[0]
            self.assertTrue(abs(len(train_idcs) - ref_train) <= 1)
            val_idcs = test_partitioner.get_file_lists_by(phase=VALID)[0]
            self.assertTrue(abs(len(val_idcs) - ref_val) <= 1
                            and not any(idx in train_idcs for idx in val_idcs))
            infer_idcs = test_partitioner.get_file_lists_by(phase=INFER)[0]
            self.assertTrue(abs(len(infer_idcs) - ref_infer) <= 1
                            and not any(idx in train_idcs for idx in infer_idcs)
                            and not any(idx in val_idcs for idx in infer_idcs))
            self.assertEqual(len(train_idcs) + len(val_idcs) + len(infer_idcs),
                             NUM_SUBJECTS)


if __name__ == '__main__':
    tf.test.main()
