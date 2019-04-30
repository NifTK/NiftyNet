# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

from niftynet.engine.signal import TRAIN, VALID, INFER, ALL
from niftynet.io.memory_image_source import MEMORY_INPUT_CALLBACK_PARAM
from niftynet.io.memory_image_sets_partitioner import \
    MEMORY_INPUT_NUM_SUBJECTS_PARAM
from niftynet.io.memory_image_source import make_input_spec
from niftynet.io.memory_image_sink import MEMORY_OUTPUT_CALLBACK_PARAM
from niftynet.io.image_endpoint_factory import ImageEndPointFactory


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

    def _configure_memory(self):
        from argparse import Namespace

        data_param = {MEMORY_INPUT_NUM_SUBJECTS_PARAM: 10,
                      'input': Namespace(pixdim=(),
                                         axcodes=(),
                                         filename_contains='some-string',
                                         interp_order=3),
                      'output': Namespace(pixdim=(),
                                          axcodes=(),
                                          filename_contains='another-string',
                                          interp_order=0)}
        data_param['input'] = make_input_spec(data_param['input'],
                                              self._input_callback1)
        data_param['output'] = make_input_spec(data_param['output'],
                                               self._input_callback2)

        action_param = Namespace(output_postfix='never-read',
                                 num_classes=2,
                                 output_interp_order=1,
                                 spatial_window_size=(80, 80))
        vars(action_param)[MEMORY_OUTPUT_CALLBACK_PARAM] = \
            self._output_callback

        app_param = Namespace(compulsory_labels=(0, 1),
                              image='input',
                              label='output')

        factory = ImageEndPointFactory()
        factory.set_params(data_param, app_param, action_param)

        return factory, data_param, app_param, action_param

    def test_infer_memeory(self):
        factory, data_param, _, _ = self._configure_memory()

        partitioner = factory.create_partitioner()
        partitioner.initialise(data_param=data_param,
                               new_partition=True,
                               ratios=(0.2, 0.1),
                               data_split_file=None)

        readers = factory.create_sources(['input'], INFER, INFER)
        self.assertEqual(len(readers), 1)

        writers = factory.create_sinks(readers)

        num_subs = data_param[MEMORY_INPUT_NUM_SUBJECTS_PARAM]
        self.assertTrue(readers[0].num_subjects > 0
                        and abs(readers[0].num_subjects - 0.1*num_subs) <= 1)

        for i in range(readers[0].num_subjects):
            idx, img_dict, interp_dict = readers[0](idx=i)

            self.assertEqual(interp_dict['input'], 3)
            self.assertEqual(i, idx)
            self.assertEqual(len(img_dict), 1)

            img = img_dict['input']

            writers[0](2*img, readers[0].get_subject_id(idx), img)

    def test_train_memory(self):
        factory, data_param, _, _ = self._configure_memory()

        partitioner = factory.create_partitioner()
        partitioner.initialise(data_param=data_param,
                               new_partition=True,
                               ratios=(0.2, 0.1),
                               data_split_file=None)

        readers = factory.create_sources(['input', 'output'], VALID, TRAIN)
        self.assertEqual(len(readers), 1)

        num_subs = data_param[MEMORY_INPUT_NUM_SUBJECTS_PARAM]
        self.assertTrue(readers[0].num_subjects > 0
                        and abs(readers[0].num_subjects - 0.2*num_subs) <= 1)

        for i in range(readers[0].num_subjects):
            idx, img_dict, interp_dict = readers[0](idx=i)

            self.assertEqual(len(img_dict), 2)
            self.assertEqual(interp_dict['input'], 3)
            self.assertEqual(interp_dict['output'], 0)
            self.assertEqual(i, idx)
            self.assertAllEqual(2*img_dict['input'], img_dict['output'])

if __name__ == '__main__':
    tf.test.main()
