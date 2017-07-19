# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import scipy.ndimage as ndimg
from scipy.ndimage.morphology import binary_fill_holes as fill_holes

from niftynet.utilities.misc_common import look_up_operations
from niftynet.utilities.misc_common import otsu_threshold
from niftynet.layer.base_layer import Layer

"""
This class defines methods to generate a binary image from an input image.
The binary image can be used as an automatic foreground selector, so that later
processing layers can only operate on the `True` locations within the image.
"""
SUPPORTED_MASK_TYPES = {'threshold_plus', 'threshold_minus',
                        'otsu_plus', 'otsu_minus', 'mean'}


class BinaryMaskingLayer(Layer):
    def __init__(self,
                 type='otsu_plus',
                 multimod_fusion='or',
                 threshold=0.0):

        super(BinaryMaskingLayer, self).__init__(name='binary_masking')
        self.type = look_up_operations(type.lower(), SUPPORTED_MASK_TYPES)
        self.multimod_fusion = multimod_fusion
        self.threshold = threshold

    def __make_mask_3d(self, image):

        assert image.ndim == 3
        mask = np.zeros_like(image, dtype=np.bool)
        thr = self.threshold
        if self.type == 'threshold_plus':
            mask[image > thr] = 1
        elif self.type == 'threshold_minus':
            mask[image < thr] = 1
        elif self.type == 'otsu_plus':
            thr = otsu_threshold(image) if \
                np.any(image) else self.threshold
            mask[image > thr] = 1
        elif self.type == 'otsu_minus':
            thr = otsu_threshold(image) if \
                np.any(image) else self.threshold
            mask[image < thr] = 1
        elif self.type == 'mean':
            thr = np.mean(image)
            mask[image > thr] = 1
        mask = ndimg.binary_dilation(mask, iterations=2)
        mask = fill_holes(mask)
        # foreground should not be empty
        assert not np.all(mask == False)
        # mask_fin = ndimg.binary_erosion(mask_bis, iterations=2)
        return mask

    def layer_op(self, image):
        if image.ndim == 3:
            return self.__make_mask_3d(image)

        if image.ndim == 5:
            mod_to_mask = [m for m in range(0, image.shape[3]) if
                           np.any(image[..., m, :])]
            mask = np.zeros_like(image, dtype=bool)
            for mod in mod_to_mask:
                for t in range(0, image.shape[4]):
                    mask[..., mod, t] = self.__make_mask_3d(image[..., mod, t])

            if self.multimod_fusion == 'or':
                for t in range(0, image.shape[4]):
                    new_mask = np.zeros(image.shape[0:3], dtype=np.bool)
                    for mod in mod_to_mask:
                        new_mask = np.logical_or(new_mask, mask[..., mod, t])
                    mask[..., t] = np.tile(np.expand_dims(new_mask, axis=-1),
                                           [1, mask.shape[3]])

            if self.multimod_fusion == 'and':
                for t in range(0, image.shape[4]):
                    new_mask = np.ones(image.shape[0:3], dtype=np.bool)
                    for mod in mod_to_mask:
                        new_mask = np.logical_and(new_mask, mask[..., mod, t])
                    mask[..., t] = np.tile(np.expand_dims(new_mask, axis=-1),
                                           [1, mask.shape[3]])
            return mask
        raise ValueError('unknown input format')
