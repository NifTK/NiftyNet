from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from layer.mean_variance_normalisation import MeanVarNormalisationLayer
from layer.binary_masking import BinaryMaskingLayer


class BinaryMaskingTEst(tf.test.TestCase):
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
            type='otsu_plus',
            multimod_fusion='or',
            threshold=0.0)
        mean_var_layer = MeanVarNormalisationLayer(
            binary_masking_func=mask_layer)
        mask_out = mean_var_layer(x, mask_layer(x))
        self.assertAllClose(x.shape, mask_out.shape)

    def test_3d_minus_shape(self):
        x = self.get_3d_input()
        mask_layer = BinaryMaskingLayer(
            type='otsu_minus',
            multimod_fusion='or',
            threshold=0.0)
        mean_var_layer = MeanVarNormalisationLayer(
            binary_masking_func=mask_layer)
        mask_out, m = mean_var_layer(x)
        self.assertAllClose(x.shape, mask_out.shape)

    def test_5d_shape(self):
        x = self.get_5d_input()
        mask_layer = BinaryMaskingLayer(
            type='threshold_minus',
            multimod_fusion='and',
            threshold=0.0)
        mean_var_layer = MeanVarNormalisationLayer(
            binary_masking_func=mask_layer)
        mask_out, m = mean_var_layer(x)
        self.assertAllClose(x.shape, mask_out.shape)

    def test_5d_mean_shape(self):
        x = self.get_5d_input()
        mask_layer = BinaryMaskingLayer(
            type='mean',
            multimod_fusion='and',
            threshold=0.0)
        mean_var_layer = MeanVarNormalisationLayer(
            binary_masking_func=mask_layer)
        mask_out = mean_var_layer(x, mask_layer(x))
        self.assertAllClose(x.shape, mask_out.shape)


if __name__ == "__main__":
    tf.test.main()
