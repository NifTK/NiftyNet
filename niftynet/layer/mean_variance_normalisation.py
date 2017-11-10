# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import numpy.ma as ma

from niftynet.layer.base_layer import Layer
from niftynet.layer.binary_masking import BinaryMaskingLayer


class MeanVarNormalisationLayer(Layer):
    """
    This class defines image-level normalisation by subtracting
    foreground mean intensity value and dividing by standard deviation
    """

    def __init__(self, image_name, binary_masking_func=None):

        self.image_name = image_name
        super(MeanVarNormalisationLayer, self).__init__(name='mean_var_norm')
        self.binary_masking_func = None
        if binary_masking_func is not None:
            assert isinstance(binary_masking_func, BinaryMaskingLayer)
            self.binary_masking_func = binary_masking_func

    def layer_op(self, image, mask=None):
        if isinstance(image, dict):
            image_data = np.asarray(image[self.image_name], dtype=np.float32)
        else:
            image_data = np.asarray(image, dtype=np.float32)

        if isinstance(mask, dict):
            image_mask = mask.get(self.image_name, None)
        elif mask is not None:
            image_mask = mask
        elif self.binary_masking_func is not None:
            image_mask = self.binary_masking_func(image_data)
        else:
            # no access to mask, default to the entire image
            image_mask = np.ones_like(image_data, dtype=np.bool)

        if image_data.ndim == 3:
            image_data = whitening_transformation(image_data, image_mask)
        if image_data.ndim == 5:
            for m in range(image_data.shape[4]):
                for t in range(image_data.shape[3]):
                    image_data[..., t, m] = whitening_transformation(
                        image_data[..., t, m], image_mask[..., t, m])

        if isinstance(image, dict):
            image[self.image_name] = image_data
            if isinstance(mask, dict):
                mask[self.image_name] = image_mask
            else:
                mask = {self.image_name: image_mask}
            return image, mask
        else:
            return image_data, image_mask


def whitening_transformation(image, mask):
    # make sure image is a monomodal volume
    masked_img = ma.masked_array(image, np.logical_not(mask))
    image = (image - masked_img.mean()) / max(masked_img.std(), 1e-5)
    return image
