# -*- coding: utf-8 -*-
import numpy as np

import utilities.misc_io as io
from .base_sampler import BaseSampler


def rand_spatial_coordinates(spatial_rank, img_size, win_size, n_samples):
    assert np.all([d >= win_size for d in img_size[:spatial_rank]])

    # consisting of starting and ending coordinates
    all_coords = np.zeros((n_samples, spatial_rank * 2), dtype=np.int)
    for i in range(0, spatial_rank):
        all_coords[:, i] = np.random.random_integers(
            0, max(img_size[i] - win_size, 1), n_samples)
        all_coords[:, i + spatial_rank] = all_coords[:, i] + win_size
    return all_coords


class UniformSampler(BaseSampler):
    """
    This class generators samples by uniformly sampling each input volume
    currently 4D input is supported, Height x Width x Depth x Modality
    """

    def __init__(self,
                 patch,
                 volume_loader,
                 patch_per_volume=1,
                 name="uniform_sampler"):

        super(UniformSampler, self).__init__(patch=patch, name=name)
        self.volume_loader = volume_loader
        self.patch_per_volume = patch_per_volume

    def layer_op(self, batch_size=1):
        """
         problems:
            check how many modalities available
            check the colon operator
            automatically handle mutlimodal by matching dims?
        """
        # batch_size is needed here so that it generates total number of
        # N samples where (N % batch_size) == 0

        spatial_rank = self.patch.spatial_rank
        while self.volume_loader.has_next:
            img, seg, weight_map, idx = self.volume_loader()

            # to make sure all volumetric data have the same spatial dims
            assert io.check_spatial_dims(spatial_rank, img, seg)
            assert io.check_spatial_dims(spatial_rank, img, weight_map)
            # match volumetric data shapes to the patch definition
            # (result will be 3d or 4d)
            img = io.match_volume_shape_to_patch_definition(
                img, self.patch.full_image_shape)
            seg = io.match_volume_shape_to_patch_definition(
                seg, self.patch.full_label_shape)
            weight_map = io.match_volume_shape_to_patch_definition(
                weight_map, self.patch.full_weight_map_shape)
            if img.ndim - spatial_rank > 1:
                raise NotImplementedError
                # time series data are not supported after this point

            # generates random spatial coordinates
            locations = rand_spatial_coordinates(spatial_rank,
                                                 img.shape,
                                                 self.patch.image_size,
                                                 self.patch_per_volume)

            for loc in locations:
                self.patch.set_data(idx, loc, img, seg, weight_map)
                yield self.patch
