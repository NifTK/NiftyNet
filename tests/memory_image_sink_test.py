from __future__ import absolute_import
"""
Unit test module for MemoryImageSource
"""

from functools import reduce, partial

import numpy as np
import tensorflow as tf

from niftynet.io.memory_image_sink import MemoryImageSink
from niftynet.io.memory_image_source import (MemoryImageSource,
                                             make_input_spec)
from niftynet.layer.pad import PadLayer
from niftynet.utilities.util_common import ParserNamespace


class MemoryImageSinkTest(tf.test.TestCase):
    def get_image(self, idx):
        return (1 + idx)*np.arange(10*100*2, dtype=np.float32)\
                           .reshape(10, 100, 1, 1, 2)

    def check_output(self, outpt, sub, inpt):
        ref = self.get_image(int(sub))

        self.assertAllEqual(ref.flatten(), inpt.flatten())
        self.assertAllEqual(outpt.flatten(), 2*inpt.flatten())

    def test_with_preprocessing(self):
        data_param = {}
        data_param['image'] = ParserNamespace(
            interp_order=1,
            pixdim=None,
            axcodes=None)

        make_input_spec(data_param['image'],
                        self.get_image,
                        do_reshape_nd=False,
                        do_typecast=False,
                        do_reshape_rgb=False)

        task_param = ParserNamespace(image=['image'])

        source = MemoryImageSource(['image'])
        source.initialise(data_param, task_param, [2, 3, 7])
        source.add_preprocessing_layers([PadLayer(['image'], [3]*2)])

        sink = MemoryImageSink(source, 1, self.check_output)

        for i in range(source.num_subjects):
            _, img_dict, _ = source(i)
            ref = self.get_image(int(source.get_subject_id(i)))

            ref_shape = [6 + d for d in ref.shape[:2]] + list(ref.shape[2:])
            self.assertAllEqual(ref_shape, list(img_dict['image'].shape))

            sink(2*img_dict['image'],
                 source.get_subject_id(i),
                 source.get_image(i)['image'])


if __name__ == '__main__':
    tf.test.main()
