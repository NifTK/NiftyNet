# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.io.image_reader import ImageReader
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.pad import PadLayer
from tests.test_util import ParserNamespace

# test multiple modalties
MULTI_MOD_DATA = {
    'T1': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1reader.csv'),
        path_to_search='testing_data',
        filename_contains=('_o_T1_time',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None
    ),
    'FLAIR': ParserNamespace(
        csv_file=os.path.join('testing_data', 'FLAIRreader.csv'),
        path_to_search='testing_data',
        filename_contains=('FLAIR_',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None
    )
}
MULTI_MOD_TASK = ParserNamespace(image=('T1', 'FLAIR'))

# test single modalities
SINGLE_MOD_DATA = {
    'lesion': ParserNamespace(
        csv_file=os.path.join('testing_data', 'lesion.csv'),
        path_to_search='testing_data',
        filename_contains=('Lesion',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None
    )
}
SINGLE_MOD_TASK = ParserNamespace(image=('lesion',))

EXISTING_DATA = {
    'lesion': ParserNamespace(
        csv_file=os.path.join('testing_data', 'lesion.csv'),
        interp_order=3,
        pixdim=None,
        axcodes=None
    )
}

# test labels
LABEL_DATA = {
    'parcellation': ParserNamespace(
        csv_file=os.path.join('testing_data', 'labels.csv'),
        path_to_search='testing_data',
        filename_contains=('Parcellation',),
        filename_not_contains=('Lesion',),
        interp_order=0,
        pixdim=(4, 4, 4),
        axcodes=None
    )
}
LABEL_TASK = ParserNamespace(label=('parcellation',))

BAD_DATA = {
    'lesion': ParserNamespace(
        csv_file=os.path.join('testing_data', 'lesion.csv'),
        path_to_search='testing_data',
        filename_contains=('Lesion',),
        filename_not_contains=('Parcellation',),
        pixdim=None,
        axcodes=None
    )
}
BAD_TASK = ParserNamespace(image=('test',))


class ImageReaderTest(tf.test.TestCase):
    def test_initialisation(self):
        with self.assertRaisesRegexp(ValueError, ''):
            reader = ImageReader(['test'])
            reader.initialise_reader(MULTI_MOD_DATA, MULTI_MOD_TASK)
        with self.assertRaisesRegexp(AssertionError, ''):
            reader = ImageReader(None)
            reader.initialise_reader(MULTI_MOD_DATA, MULTI_MOD_TASK)
        reader = ImageReader(['image'])
        reader.initialise_reader(MULTI_MOD_DATA, MULTI_MOD_TASK)
        self.assertEquals(len(reader.output_list), 4)

        reader = ImageReader(['image'])
        reader.initialise_reader(SINGLE_MOD_DATA, SINGLE_MOD_TASK)
        self.assertEquals(len(reader.output_list), 4)

    def test_properties(self):
        reader = ImageReader(['image'])
        reader.initialise_reader(SINGLE_MOD_DATA, SINGLE_MOD_TASK)
        self.assertEquals(len(reader.output_list), 4)
        self.assertDictEqual(reader.shapes,
                             {'image': (256, 168, 256, 1, 1)})
        self.assertDictEqual(reader.tf_dtypes, {'image': tf.float32})
        self.assertEqual(reader.names, ['image'])
        self.assertDictEqual(reader.input_sources,
                             {'image': ('lesion',)})
        self.assertEqual(reader.get_subject_id(1)[:4], 'Fin_')

    def test_existing_csv(self):
        reader_for_csv = ImageReader(['image'])
        reader_for_csv.initialise_reader(SINGLE_MOD_DATA, SINGLE_MOD_TASK)
        reader = ImageReader(['image'])
        reader.initialise_reader(EXISTING_DATA, SINGLE_MOD_TASK)
        self.assertEquals(len(reader.output_list), 4)
        self.assertDictEqual(reader.shapes,
                             {'image': (256, 168, 256, 1, 1)})
        self.assertDictEqual(reader.tf_dtypes, {'image': tf.float32})
        self.assertEqual(reader.names, ['image'])
        self.assertDictEqual(reader.input_sources,
                             {'image': ('lesion',)})
        self.assertEqual(reader.get_subject_id(1)[:4], 'Fin_')

    def test_operations(self):
        reader = ImageReader(['image'])
        reader.initialise_reader(SINGLE_MOD_DATA, SINGLE_MOD_TASK)
        idx, data, interp_order = reader()
        self.assertEqual(SINGLE_MOD_DATA['lesion'].interp_order,
                         interp_order['image'][0])
        self.assertAllClose(data['image'].shape, (256, 168, 256, 1, 1))

    def test_preprocessing(self):
        reader = ImageReader(['image'])
        reader.initialise_reader(SINGLE_MOD_DATA, SINGLE_MOD_TASK)
        idx, data, interp_order = reader()
        self.assertEqual(SINGLE_MOD_DATA['lesion'].interp_order,
                         interp_order['image'][0])
        self.assertAllClose(data['image'].shape, (256, 168, 256, 1, 1))
        reader.add_preprocessing_layers(
            [PadLayer(image_name=['image'], border=(10, 5, 5))])
        idx, data, interp_order = reader(idx=2)
        self.assertEqual(idx, 2)
        self.assertAllClose(data['image'].shape, (276, 178, 266, 1, 1))

    def test_trainable_preprocessing(self):
        label_file = os.path.join('testing_data', 'label_reader.txt')
        if os.path.exists(label_file):
            os.remove(label_file)
        label_normaliser = DiscreteLabelNormalisationLayer(
            image_name='label',
            modalities=vars(LABEL_TASK).get('label'),
            model_filename=os.path.join('testing_data', 'label_reader.txt'))
        reader = ImageReader(['label'])
        with self.assertRaisesRegexp(AssertionError, ''):
            reader.add_preprocessing_layers(label_normaliser)
        reader.initialise_reader(LABEL_DATA, LABEL_TASK)
        reader.add_preprocessing_layers(label_normaliser)
        reader.add_preprocessing_layers(
            [PadLayer(image_name=['label'], border=(10, 5, 5))])
        idx, data, interp_order = reader(idx=0)
        unique_data = np.unique(data['label'])
        expected = np.array(range(156), dtype=np.float32)
        self.assertAllClose(unique_data, expected)
        self.assertAllClose(data['label'].shape, (83, 73, 73, 1, 1))

    def test_errors(self):
        with self.assertRaisesRegexp(AttributeError, ''):
            reader = ImageReader(['image'])
            reader.initialise_reader(BAD_DATA, SINGLE_MOD_TASK)
        with self.assertRaisesRegexp(ValueError, ''):
            reader = ImageReader(['image'])
            reader.initialise_reader(SINGLE_MOD_DATA, BAD_TASK)

        reader = ImageReader(['image'])
        reader.initialise_reader(SINGLE_MOD_DATA, SINGLE_MOD_TASK)
        idx, data, interp_order = reader(idx=100)
        self.assertEqual(idx, -1)
        self.assertEqual(data, None)
        idx, data, interp_order = reader(shuffle=True)
        self.assertEqual(data['image'].shape, (256, 168, 256, 1, 1))


if __name__ == "__main__":
    tf.test.main()
