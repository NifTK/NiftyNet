from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.upsample import UpSampleLayer
from tests.niftynet_testcase import NiftyNetTestCase

class UpSampleTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (4, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (4, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def _test_upsample_shape(self, rank, param_dict, output_shape):
        if rank == 3:
            input_data = self.get_3d_input()
        elif rank == 2:
            input_data = self.get_2d_input()

        upsample_layer = UpSampleLayer(**param_dict)
        output_data = upsample_layer(input_data)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(output_data)
            self.assertAllClose(output_shape, output.shape)

    def test_3d_default_replicate(self):
        input_param = {'func': 'REPLICATE',
                       'kernel_size': 3,
                       'stride': 3}
        self._test_upsample_shape(rank=3,
                                  param_dict=input_param,
                                  output_shape=(4, 48, 48, 48, 8))

    def test_3d_default_channelwise_deconv(self):
        input_param = {'func': 'CHANNELWISE_DECONV',
                       'kernel_size': 3,
                       'stride': 3}
        self._test_upsample_shape(rank=3,
                                  param_dict=input_param,
                                  output_shape=(4, 48, 48, 48, 8))

    def test_3d_replicate(self):
        input_param = {'func': 'REPLICATE',
                       'kernel_size': [3, 1, 3],
                       'stride': [3, 1, 3]}
        self._test_upsample_shape(rank=3,
                                  param_dict=input_param,
                                  output_shape=(4, 48, 16, 48, 8))

    def test_3d_channelwise_deconv(self):
        input_param = {'func': 'CHANNELWISE_DECONV',
                       'kernel_size': [1, 3, 2],
                       'stride': [1, 2, 3]}
        self._test_upsample_shape(rank=3,
                                  param_dict=input_param,
                                  output_shape=(4, 16, 32, 48, 8))

    def test_2d_default_replicate(self):
        input_param = {'func': 'REPLICATE',
                       'kernel_size': 3,
                       'stride': 3}
        self._test_upsample_shape(rank=2,
                                  param_dict=input_param,
                                  output_shape=(4, 48, 48, 8))

    def test_2d_default_channelwise_deconv(self):
        input_param = {'func': 'CHANNELWISE_DECONV',
                       'kernel_size': 3,
                       'stride': 3}
        self._test_upsample_shape(rank=2,
                                  param_dict=input_param,
                                  output_shape=(4, 48, 48, 8))

    def test_2d_replicate(self):
        input_param = {'func': 'REPLICATE',
                       'kernel_size': [3, 1],
                       'stride': [3, 1]}
        self._test_upsample_shape(rank=2,
                                  param_dict=input_param,
                                  output_shape=(4, 48, 16, 8))

    def test_2d_channelwise_deconv(self):
        input_param = {'func': 'CHANNELWISE_DECONV',
                       'kernel_size': [1, 3],
                       'stride': [1, 2]}
        self._test_upsample_shape(rank=2,
                                  param_dict=input_param,
                                  output_shape=(4, 16, 32, 8))


if __name__ == "__main__":
    tf.test.main()
