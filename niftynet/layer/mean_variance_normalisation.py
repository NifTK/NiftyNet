# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import numpy.ma as ma

from niftynet.layer.base_layer import Layer
from niftynet.layer.binary_masking import BinaryMaskingLayer

"""
This class defines image-level normalisation by subtracting
foreground mean intensity value and dividing by standard deviation
"""


class MeanVarNormalisationLayer(Layer):
    def __init__(self, binary_masking_func=None):

        layer_name = 'mean_var_norm'
        super(MeanVarNormalisationLayer, self).__init__(name=layer_name)
        if binary_masking_func is not None:
            assert isinstance(binary_masking_func, BinaryMaskingLayer)
            self.binary_masking_func = binary_masking_func

    def __whitening_transformation_3d(self, image, mask):
        # make sure image is a monomodal volume
        assert image.ndim == 3

        masked_img = ma.masked_array(np.copy(image), np.logical_not(mask))
        mean = masked_img.mean()
        std = masked_img.std()
        image -= mean
        image /= max(std, 1e-5)
        return image

    def layer_op(self, image, mask=None):
        image = np.asarray(image, dtype=float)

        image_mask = None
        if mask is not None:
            image_mask = np.asarray(mask, dtype=np.bool)
        else:
            if self.binary_masking_func is not None:
                image_mask = self.binary_masking_func(image)

        # no access to mask, default to all foreground
        if image_mask is None:
            image_mask = np.ones_like(image, dtype=np.bool)

        if image.ndim == 3:
            image = self.__whitening_transformation_3d(image, image_mask)

        if image.ndim == 5:
            for m in range(0, image.shape[3]):
                for t in range(0, image.shape[4]):
                    image[..., m, t] = self.__whitening_transformation_3d(
                        image[..., m, t], image_mask[..., m, t])

        return image, image_mask
