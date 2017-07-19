# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from copy import deepcopy

import math
import numpy as np
import scipy.ndimage
import niftynet.utilities.misc_io as io
from niftynet.engine.base_sampler import BaseSampler


class ResizeSampler(BaseSampler):
    """
    This class generates samples by rescaling the whole image to the desired size
    currently 4D input is supported, Height x Width x Depth x Modality
    """

    def __init__(self,
                 patch,
                 volume_loader,
                 data_augmentation_methods=None,
                 name="resize_sampler"):

        super(ResizeSampler, self).__init__(patch=patch, name=name)
        self.volume_loader = volume_loader
        if data_augmentation_methods is None:
            self.data_augmentation_layers = []
        else:
            self.data_augmentation_layers = data_augmentation_methods

    def layer_op(self, batch_size=1):
        """
         problems:
            check how many modalities available
            check the colon operator
            automatically handle mutlimodal by matching dims?
        """
        spatial_rank = self.patch.spatial_rank
        local_layers = [deepcopy(x) for x in self.data_augmentation_layers]
        patch = deepcopy(self.patch)
        while self.volume_loader.has_next:
            img, seg, weight_map, idx = self.volume_loader()
            # to make sure all volumetric data have the same spatial dims
            # and match volumetric data shapes to the patch definition
            # (the matched result will be either 3d or 4d)
            img.spatial_rank = spatial_rank

            img.data = io.match_volume_shape_to_patch_definition(
                img.data, patch)
            if img.data.ndim == 5:
                raise NotImplementedError
                # time series data are not supported
            if seg is not None:
                seg.spatial_rank = spatial_rank
                seg.data = io.match_volume_shape_to_patch_definition(
                    seg.data, patch)
            if weight_map is not None:
                weight_map.spatial_rank = spatial_rank
                weight_map.data = io.match_volume_shape_to_patch_definition(
                    weight_map.data, patch)

            # apply volume level augmentation
            for aug in local_layers:
                aug.randomise(spatial_rank=spatial_rank)
                img, seg, weight_map = aug(img), aug(seg), aug(weight_map)
            # resize image to patch size
            i_spatial_rank=int(math.ceil(spatial_rank))
            zoom=[p/d for p,d in zip([patch.image_size]*i_spatial_rank,img.data.shape)]+[1]
            
            img = scipy.ndimage.interpolation.zoom(img.data, zoom, order=img.interp_order)
            if seg is not None:
              seg = scipy.ndimage.interpolation.zoom(seg.data, zoom, order=seg.interp_order)
            if weight_map is not None:
              weight_map = scipy.ndimage.interpolation.zoom(weight_map.data, zoom, order=weight_map.interp_order)
            loc=[0]*i_spatial_rank+[patch.image_size]*i_spatial_rank
            patch.set_data(idx, loc, img, seg, weight_map)
            yield patch
