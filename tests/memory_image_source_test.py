from __future__ import absolute_import
"""
Unit test module for MemoryImageSource
"""

from functools import reduce, partial

import numpy as np
import tensorflow as tf

from niftynet.io.memory_image_source import (MemoryImageSource,
                                             make_input_spec)
from niftynet.utilities.util_common import ParserNamespace

NOF_IMAGES = 11
BASE_IDX = 10


def _expected_shape(idx):
    return (10, 10 + idx, 10 + 2*idx, 2)


def _test_image_callback(idx, sidx, dtype):
    """
    Test image input callback

    :param idx: image index
    :param sidx: source index
    :param dtype: output data type
    """

    shape = _expected_shape(idx)
    size = reduce(lambda prod, s: prod*s, shape, 1)

    return (sidx + 1)*np.arange(size).reshape(_expected_shape(idx)).astype(dtype)


class MemoryImageSourceTest(tf.test.TestCase):
    def test_multi_source(self):
        interps = (0, 1, 3)
        dtypes = (np.float32, np.uint32, np.int32)
        nof_sources = 3
        data_param = {}

        def _get_source_name(sidx):
            return 'image%i' % sidx

        for sidx in range(nof_sources):
            name = _get_source_name(sidx)
            interp = interps[sidx % len(interps)]
            dtype = dtypes[sidx % len(dtypes)]
            data_param[name] = ParserNamespace(
                interp_order=interp,
                pixdim=None,
                axcodes=None)

            kwargs = {'do_reshape_nd': True}
            if dtype != np.float32:
                kwargs['do_typecast'] = True

            source = partial(_test_image_callback, sidx=sidx, dtype=dtype)
            make_input_spec(data_param[name], source, **kwargs)

        kwargs = {key + '_1': [key] for key in data_param}
        task_param = ParserNamespace(**kwargs)

        source = MemoryImageSource(list(kwargs.keys()))
        source.initialise(data_param, task_param,
                          list(range(BASE_IDX, BASE_IDX + NOF_IMAGES)))

        for idx in range(NOF_IMAGES):
            int_idx, image_dict, interp_dict = source(idx=idx)

            self.assertEqual(int_idx, idx)
            self.assertEqual(len(image_dict), nof_sources)
            self.assertEqual(len(interp_dict), nof_sources)

            for name in image_dict:
                img = image_dict[name]

                sidx = int(name[-3])
                assert _get_source_name(sidx) + '_1' == name

                interp = interps[sidx % len(interps)]
                self.assertEqual(len(interp_dict[name]), 1)
                self.assertEqual(interp, interp_dict[name][0])

                expected = _test_image_callback(idx + BASE_IDX, sidx, np.float32)
                expected_shape \
                    = list(expected.shape[:3]) + [1] + [expected.shape[-1]]
                expected = expected.reshape(expected_shape)

                self.assertEqual(img.dtype, np.float32)
                self.assertAllEqual(img.shape, expected_shape)
                self.assertAllClose(img.flatten(), expected.flatten())


if __name__ == '__main__':
    tf.test.main()
