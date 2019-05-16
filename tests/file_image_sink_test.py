# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import csv
import os

import nibabel as nib
import tensorflow as tf

from niftynet.io.file_image_sink import FileImageSink
from tests.windows_aggregator_grid_v2_test import (get_3d_reader,
                                                   get_2d_reader,
                                                   get_label_reader)

OUTPUT_PATH = os.path.join('testing_data', 'aggregated')
POSTFIX = '_writer_test_out'


def get_writer(reader):
    return FileImageSink(
        reader, 0, output_path=OUTPUT_PATH, postfix=POSTFIX)


class FileImageSinkTest(tf.test.TestCase):
    def _test_content(self, sub, writer):
        self.assertTrue(os.path.isfile(os.path.join(
            writer.output_path, sub + writer.postfix + '.nii.gz')))

    def _test_logging(self, sub_list):
        files = {sub: None for sub in sub_list}
        with open(os.path.join(OUTPUT_PATH, 'inferred.csv'), 'r') as fin:
            for row in csv.reader(fin):
                self.assertTrue(row[0] in sub_list)
                files[row[0]] = row[1]

        self.assertTrue(all(not files[sub] is None for sub in sub_list))
        for file in files.values():
            self.assertTrue(os.path.isfile(file))

    def _test_writer(self, reader):
        writer = get_writer(reader)
        self.assertEqual(writer.postfix, POSTFIX)
        self.assertEqual(os.path.abspath(writer.output_path),
                         os.path.abspath(OUTPUT_PATH))

        subs = []
        for idx in range(reader.num_subjects):
            _, inpt, _ = reader(idx=idx)
            sub = reader.get_subject_id(idx)
            outpt = (1 + idx)*inpt['image']

            writer(outpt, sub, reader.get_output_image(idx)['image'])
            self._test_content(sub, writer)
            subs.append(sub)

        self._test_logging(subs)


    def test_3d(self):
        self._test_writer(get_3d_reader())


    def test_2d(self):
        self._test_writer(get_2d_reader())

    def test_preprocessing(self):
        reader = get_label_reader()
        writer = get_writer(reader)

        for idx in range(reader.num_subjects):
            _, inpt, _ = reader(idx=idx)
            sub = reader.get_subject_id(idx)
            outpt = (1 + idx)*inpt['label']

            ref = reader.get_output_image(idx)['label']
            writer(outpt, sub, ref)
            img = nib.load(os.path.join(
                OUTPUT_PATH, sub + writer.postfix + '.nii.gz')).get_data()
            self.assertEqual(img.shape, ref.shape)


if __name__ == "__main__":
    tf.test.main()
