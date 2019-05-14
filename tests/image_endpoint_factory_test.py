# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import tensorflow as tf
import nibabel as nib
import numpy as np

from niftynet.engine.signal import TRAIN, VALID, INFER, ALL
from niftynet.io.memory_image_source import MEMORY_INPUT_CALLBACK_PARAM
from niftynet.io.memory_image_sets_partitioner import \
    set_number_of_memory_subjects
from niftynet.io.memory_image_source import make_input_spec
from niftynet.io.memory_image_sink import make_output_spec
from niftynet.io.image_endpoint_factory import ImageEndPointFactory
from niftynet.utilities.util_common import ParserNamespace


class ImageEndPointFactoryTest(tf.test.TestCase):
    @staticmethod
    def _input_callback1(idx):
        return np.full((100, 100, 1, 1, 1), idx)

    @staticmethod
    def _input_callback2(idx):
        return np.full((100, 100, 1, 1, 1), 2*idx)

    def _output_callback(self, image_out, sub, image_data_in):
        self.assertAllEqual(image_out.shape, image_data_in.shape)
        self.assertTrue((image_data_in == int(sub)).sum() == image_data_in.size)
        self.assertTrue((image_out == 2*int(sub)).sum() == image_data_in.size)

    NUM_MEM_SUBJECTS = 10

    def _configure_memory(self):
        data_param = {'input': ParserNamespace(pixdim=(),
                                               axcodes=(),
                                               filename_contains='some-string',
                                               interp_order=3),
                      'output': ParserNamespace(pixdim=(),
                                                axcodes=(),
                                                filename_contains='another-string',
                                                interp_order=0)}
        data_param['input'] = make_input_spec(data_param['input'],
                                              self._input_callback1)
        data_param['output'] = make_input_spec(data_param['output'],
                                               self._input_callback2)
        set_number_of_memory_subjects(data_param, self.NUM_MEM_SUBJECTS)

        action_param = ParserNamespace(output_postfix='never-read',
                                       num_classes=2,
                                       output_interp_order=1,
                                       spatial_window_size=(80, 80))
        make_output_spec(action_param, self._output_callback)

        app_param = ParserNamespace(compulsory_labels=(0, 1),
                                    image=('input',),
                                    label=('output',))

        factory = ImageEndPointFactory()
        factory.set_params(data_param, app_param, action_param)

        return factory, data_param, app_param, action_param

    def test_infer_memory(self):
        factory, data_param, _, _ = self._configure_memory()

        partitioner = factory.create_partitioner()
        partitioner.initialise(data_param=data_param,
                               new_partition=True,
                               ratios=(0.2, 0.1),
                               data_split_file=None)

        readers = factory.create_sources(['image'], INFER, INFER)
        self.assertEqual(len(readers), 1)

        writers = factory.create_sinks(readers)

        num_subs = partitioner.number_of_subjects()
        self.assertEqual(num_subs, self.NUM_MEM_SUBJECTS)
        self.assertTrue(readers[0].num_subjects > 0
                        and abs(readers[0].num_subjects - 0.1*num_subs) <= 1)

        for i in range(readers[0].num_subjects):
            idx, img_dict, interp_dict = readers[0](idx=i)

            self.assertEqual(interp_dict['image'][0], 3)
            self.assertEqual(i, idx)
            self.assertEqual(len(img_dict), 1)

            img = img_dict['image']

            writers[0](2*img, readers[0].get_subject_id(idx), img)

    def test_train_memory(self):
        factory, data_param, _, _ = self._configure_memory()

        partitioner = factory.create_partitioner()
        partitioner.initialise(data_param=data_param,
                               new_partition=True,
                               ratios=(0.2, 0.1),
                               data_split_file=None)

        readers = factory.create_sources(['image', 'label'], VALID, TRAIN)
        self.assertEqual(len(readers), 1)

        num_subs = partitioner.number_of_subjects()
        self.assertTrue(readers[0].num_subjects > 0
                        and abs(readers[0].num_subjects - 0.2*num_subs) <= 1)

        for i in range(readers[0].num_subjects):
            idx, img_dict, interp_dict = readers[0](idx=i)

            self.assertEqual(len(img_dict), 2)
            self.assertEqual(interp_dict['image'][0], 3)
            self.assertEqual(interp_dict['label'][0], 0)
            self.assertEqual(i, idx)
            self.assertAllEqual(2*img_dict['image'], img_dict['label'])

    def _configure_file(self):
        from tests.file_image_sets_partitioner_test import test_sections

        data_param = {'input': test_sections['T1'],
                      'output': test_sections['Flair']}

        action_param = ParserNamespace(output_postfix='_file_output',
                                       num_classes=2,
                                       save_seg_dir=os.path.join(
                                           'testing_data', 'aggregated'),
                                       output_interp_order=1,
                                       spatial_window_size=(80, 80))
        make_output_spec(action_param, self._output_callback)

        app_param = ParserNamespace(compulsory_labels=(0, 1),
                                    image=('input',),
                                    label=('output',))

        factory = ImageEndPointFactory()
        factory.set_params(data_param, app_param, action_param)

        return factory, data_param, app_param, action_param

    def test_file_infer(self):
        factory, data_param, _, action_param = self._configure_file()

        print('data', data_param)
        print('act', action_param)

        partitioner = factory.create_partitioner()

        partitioner.initialise(data_param=data_param,
                               new_partition=True,
                               ratios=(0.0, 0.8),
                               data_split_file=None)

        num_subs = partitioner.number_of_subjects()

        readers = factory.create_sources(['image'], INFER, INFER)
        self.assertEqual(len(readers), 1)

        writers = factory.create_sinks(readers)

        self.assertGreater(num_subs, 1)
        self.assertGreater(readers[0].num_subjects, 1)

        def _get_destination_path(sub):
            return os.path.join(action_param.save_seg_dir, '{}{}.nii.gz'.format(
                sub, action_param.output_postfix))

        for i in range(readers[0].num_subjects):
            idx, images, interps = readers[0](idx=i)

            self.assertEqual(i, idx)
            self.assertEqual(interps['image'][0],
                             data_param['input'].interp_order)

            out = images['image']*2
            sub = readers[0].get_subject_id(idx)
            img = readers[0].output_list[idx]['image']
            writers[0](out, sub, img)

            reloaded = nib.load(_get_destination_path(sub)).get_data()

            self.assertAllClose(reloaded.flatten(), out.flatten(), rtol=1e-3)


if __name__ == '__main__':
    tf.test.main()
