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
    def __init__(self, field, binary_masking_func=None):

        self.field = field
        super(MeanVarNormalisationLayer, self).__init__(name='mean_var_norm')
        if binary_masking_func is not None:
            assert isinstance(binary_masking_func, BinaryMaskingLayer)
            self.binary_masking_func = binary_masking_func

    def layer_op(self, image, mask=None):
        if isinstance(image, dict):
            image_data = np.asarray(image[self.field], dtype=np.float32)
        else:
            image_data = np.asarray(image, dtype=np.float32)

        image_mask = None
        if isinstance(mask, dict):
            image_mask = mask.get(self.field, None)
        elif mask is not None:
            image_mask = mask
        elif self.binary_masking_func is not None:
            image_mask = self.binary_masking_func(image_5d)
        else:
            # no access to mask, default to the entire image
            image_mask = np.ones_like(image_5d, dtype=np.bool)

        if image_data.ndim == 3:
            image_data = whitening_transformation(image_data, image_mask)
        if image_data.ndim == 5:
            for m in range(0, image_data.shape[4]):
                image_data[..., m] = whitening_transformation(
                    image_data[..., m], image_mask[..., m])

        if isinstance(image, dict):
            image[self.field] = image_data
            if isinstance(mask, dict):
                mask[self.field] = image_mask
            else:
                mask = {self.field: image_mask}
            return image, mask
        else:
            return image_data, image_mask


def whitening_transformation(image, mask):
    # make sure image is a monomodal volume
    masked_img = ma.masked_array(np.copy(image), np.logical_not(mask))
    mean = masked_img.mean()
    std = masked_img.std()
    image -= mean
    image /= max(std, 1e-5)
    return image
