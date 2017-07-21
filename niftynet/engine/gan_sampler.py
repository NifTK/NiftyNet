# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from copy import deepcopy

import math
import numpy as np
import scipy.ndimage
import niftynet.utilities.misc_io as io
from niftynet.engine.base_sampler import BaseSampler
import warnings

class GANSampler(BaseSampler):
    """
    This class generates samples by rescaling the whole image to the desired size
    currently 4D input is supported, Height x Width x Depth x Modality
    """

    def __init__(self,
                 patch,
                 volume_loader,
                 patch_per_volume=1,
                 data_augmentation_methods=None,
                 name="gan_sampler"):

        super(GANSampler, self).__init__(patch=patch, name=name)
        self.volume_loader = volume_loader
        if patch.spatial_rank==2.5:
          self.patch_per_volume = patch_per_volume
        else:
          if patch_per_volume!=1:
            warnings.warn('GANSampler cannot sample more than one path with spatial_rank={}'.format(patch.spatial_rank))
          self.patch_per_volume=1
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
            zoom = [p/d for p,d in zip([patch.image_size]*3,img.data.shape[:3])]+[1]
            # resize image to patch size
            noise = np.random.randn(patch.noise_size)
            if spatial_rank==3:
                spatial_rank = 3
                loc=[0]*spatial_rank+[patch.image_size]*spatial_rank
                img = scipy.ndimage.interpolation.zoom(img.data, zoom)
                patch.set_data(idx, loc, img, cond, noise)
                yield patch
            elif spatial_rank==2.5:
                loc=[0,0,0]+[patch.image_size]*2
                for it in range(self.patch_per_volume):
                    slice = np.random.randint(0, img.data.shape[2],1)
                    img = np.expand_dims(scipy.ndimage.interpolation.zoom(img.data[:,:,slice[0],:], zoom[:2]+zoom[-1:]),2)
                    patch.set_data(idx, loc, img, cond, noise)
                    yield patch
                    noise = np.random.randn(patch.noise_size)
            elif spatial_rank==2:
                loc=[0,0]+[patch.image_size]*2
                img = np.expand_dims(scipy.ndimage.interpolation.zoom(img.data[:,:,0,:], zoom[:2]+zoom[-1:]),2)
                patch.set_data(idx, loc, img, cond, noise)
                yield patch
