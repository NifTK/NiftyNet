# -*- coding: utf-8 -*-
import numpy as np

import utilities.misc_io as io
from .base_sampler import BaseSampler
from .rand_rotation import RandomRotationLayer


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
                 data_augmentation_methods=['rotation'],
                 name="uniform_sampler"):

        super(UniformSampler, self).__init__(patch=patch, name=name)
        self.volume_loader = volume_loader
        self.patch_per_volume = patch_per_volume
        self.data_augmentation_layers = []
        for method in data_augmentation_methods:
            if method == 'rotation':
                self.data_augmentation_layers.append(
                    RandomRotationLayer(min_angle=-10.0, max_angle=10.0))
            else:
                raise ValueError('unkown data augmentation method')

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
        param_dict = {'spatial_rank': self.patch.spatial_rank,
                      'interp_order': self.volume_loader.interp_order}
        while self.volume_loader.has_next:
            img, seg, weight_map, idx = self.volume_loader()

            # to make sure all volumetric data have the same spatial dims
            # and match volumetric data shapes to the patch definition
            # (the matched result will be either 3d or 4d)
            img.spatial_rank = spatial_rank
            img.data = io.match_volume_shape_to_patch_definition(
                img.data, self.patch.full_image_shape)
            if img.data.ndim - spatial_rank > 1:
                raise NotImplementedError
                # time series data are not supported
            if seg is not None:
                seg.spatial_rank = spatial_rank
                seg.data = io.match_volume_shape_to_patch_definition(
                    seg.data, self.patch.full_label_shape)
            if weight_map is not None:
                weight_map.spatial_rank = spatial_rank
                weight_map.data = io.match_volume_shape_to_patch_definition(
                    weight_map.data, self.patch.full_weight_map_shape)

            # apply volume level augmentation
            for layer in self.data_augmentation_layers:
                layer.randomise()
                img, seg, weight_map = layer(img), layer(seg), layer(weight_map)

            # generates random spatial coordinates
            locations = rand_spatial_coordinates(img.spatial_rank,
                                                 img.data.shape,
                                                 self.patch.image_size,
                                                 self.patch_per_volume)

            for loc in locations:
                self.patch.set_data(idx, loc, img, seg, weight_map)
                yield self.patch
