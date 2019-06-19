from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.binary_masking import BinaryMaskingLayer
from tests.niftynet_testcase import NiftyNetTestCase


class BinaryMaskingTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (16, 16, 16)
        x = np.random.randint(-10, 10, size=input_shape)
        return x

    def get_5d_input(self):
        input_shape = (16, 16, 16, 3, 2)
        x = np.random.randint(-10, 10, size=input_shape)
        return x

    def test_3d_plus_shape(self):
        x = self.get_3d_input()
        mask_layer = BinaryMaskingLayer(
            type_str='otsu_plus',
            multimod_fusion='or',
            threshold=0.0)
        mask_out = mask_layer(x)
        print(mask_layer)
        self.assertAllClose(x.shape, mask_out.shape)

    def test_3d_minus_shape(self):
        x = self.get_3d_input()
        mask_layer = BinaryMaskingLayer(
            type_str='otsu_minus',
            multimod_fusion='or',
            threshold=0.0)
        mask_out = mask_layer(x)
        print(mask_layer)
        self.assertAllClose(x.shape, mask_out.shape)

    def test_5d_shape(self):
        x = self.get_5d_input()
        mask_layer = BinaryMaskingLayer(
            type_str='threshold_minus',
            multimod_fusion='and',
            threshold=0.0)
        mask_out = mask_layer(x)
        print(mask_layer)
        self.assertAllClose(x.shape, mask_out.shape)

    def test_5d_mean_shape(self):
        x = self.get_5d_input()
        mask_layer = BinaryMaskingLayer(
            type_str='mean_plus',
            multimod_fusion='and',
            threshold=0.0)
        mask_out = mask_layer(x)
        print(mask_layer)
        self.assertAllClose(x.shape, mask_out.shape)


if __name__ == "__main__":
    tf.test.main()
