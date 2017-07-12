# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from copy import deepcopy

import math
import numpy as np
import scipy.ndimage
import niftynet.utilities.misc_io as io
from niftynet.engine.base_sampler import BaseSampler


class GANSampler(BaseSampler):
    """
    This class generates samples by rescaling the whole image to the desired size
    currently 4D input is supported, Height x Width x Depth x Modality
    """

    def __init__(self,
                 patch,
                 volume_loader,
                 data_augmentation_methods=None,
                 name="gan_sampler"):

        super(GANSampler, self).__init__(patch=patch, name=name)
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
            img, cond, weight_map, idx = self.volume_loader()
            # to make sure all volumetric data have the same spatial dims
            # and match volumetric data shapes to the patch definition
            # (the matched result will be either 3d or 4d)
            img.spatial_rank = spatial_rank

            img.data = io.match_volume_shape_to_patch_definition(
                img.data, patch)
            if img.data.ndim == 5:
                raise NotImplementedError
                # time series data are not supported
            if cond is not None:
                cond.spatial_rank = spatial_rank
                cond.data = io.match_volume_shape_to_patch_definition(
                    cond.data, patch)

            # apply volume level augmentation
            for aug in local_layers:
                aug.randomise(spatial_rank=spatial_rank)
                img, seg, weight_map = aug(img), aug(seg), aug(weight_map)
            # resize image to patch size
            i_spatial_rank=int(math.ceil(spatial_rank))
            zoom=[p/d for p,d in zip([patch.image_size]*i_spatial_rank,img.data.shape)]+[1]
            
            img = scipy.ndimage.interpolation.zoom(img.data, zoom)
            loc=[0]*i_spatial_rank+[patch.image_size]*i_spatial_rank
            noise = np.random.randn(patch.noise_size)
            patch.set_data(idx, loc, img, cond, noise)
            yield patch
