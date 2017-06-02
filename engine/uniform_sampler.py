# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np

import utilities.misc_io as io
from engine.base_sampler import BaseSampler


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
    This class generates samples by uniformly sampling each input volume,
    currently up to 4D input is supported, Height x Width x Depth x Modality
    """
    def __init__(self,
                 patch,
                 volume_loader,
                 patch_per_volume=1,
                 data_augmentation_methods=None,
                 name="uniform_sampler"):

        super(UniformSampler, self).__init__(patch=patch, name=name)
        self.volume_loader = volume_loader
        self.patch_per_volume = patch_per_volume
        if data_augmentation_methods is None:
            self.data_augmentation_layers = []
        else:
            self.data_augmentation_layers = data_augmentation_methods

    def layer_op(self, batch_size=1):
        """
        This layer reads the input volume size and generate
        randomised sampling coordinates, patches at the coordinates
        will be loaded and sent to the buffer.
        This operation can be called from multiple threads
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
                img.data, patch.full_image_shape)
            if img.data.ndim - spatial_rank > 1:
                raise NotImplementedError
                # time series data are not supported
            if seg is not None:
                seg.spatial_rank = spatial_rank
                seg.data = io.match_volume_shape_to_patch_definition(
                    seg.data, patch.full_label_shape)
            if weight_map is not None:
                weight_map.spatial_rank = spatial_rank
                weight_map.data = io.match_volume_shape_to_patch_definition(
                    weight_map.data, patch.full_weight_map_shape)

            # apply volume level augmentation
            for aug in local_layers:
                aug.randomise(spatial_rank=spatial_rank)
                img, seg, weight_map = aug(img), aug(seg), aug(weight_map)

            # generates random spatial coordinates
            locations = rand_spatial_coordinates(img.spatial_rank,
                                                 img.data.shape,
                                                 patch.image_size,
                                                 self.patch_per_volume)
            for loc in locations:
                patch.set_data(idx, loc, img, seg, weight_map)
                yield patch
