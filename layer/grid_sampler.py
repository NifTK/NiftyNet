# -*- coding: utf-8 -*-
import numpy as np

import utilities.misc_io as io
from .base_sampler import BaseSampler


def generate_grid_coordinates(spatial_rank, img_size, win_size, grid_size):
    if grid_size <= 0:
        return None
    assert np.all([d >= win_size for d in img_size[:spatial_rank]])

    # generating sampling points along each dim
    steps_along_each = [_enumerate_step_points(0,
                                               img_size[i],
                                               win_size,
                                               grid_size)
                        for i in range(0, spatial_rank)]
    # create a mesh grid
    starting_ = np.asarray(np.meshgrid(*steps_along_each))
    starting_ = starting_.reshape((spatial_rank, -1))
    # transform mesh grid into a list of coordinates
    all_coordinates = np.zeros((starting_.shape[1], spatial_rank * 2),
                               dtype=np.int)
    for i in range(0, spatial_rank):
        all_coordinates[:, i] = starting_[i, :]
        all_coordinates[:, i + spatial_rank] = starting_[i, :] + win_size
    return all_coordinates


def _enumerate_step_points(starting, ending, win_size, step_size):
    sampling_point_set = []
    while (starting + win_size) <= ending:
        sampling_point_set.append(starting)
        starting = starting + step_size
    sampling_point_set.append(np.max((ending - win_size, 0)))
    return np.unique(sampling_point_set).flatten()


class GridSampler(BaseSampler):
    """
    This class generators samples from a fixed sampling grid
    currently 4D input is supported, Hight x Width x Depth x Modality
    """

    def __init__(self, patch, volume_loader, grid_size=1, name="grid_sampler"):
        super(GridSampler, self).__init__(patch=patch, name=name)
        self.volume_loader = volume_loader
        self.grid_size = grid_size

        # this sampler is used for inference only, should not shuffle the input
        assert not self.volume_loader.is_training

    def layer_op(self, batch_size=1):
        """
        assumes img, seg the same size
        this function should be called with only one thread at a time
        """
        # batch_size is needed here so that it generates total number of
        # N samples where (N % batch_size) == 0

        spatial_rank = self.patch.spatial_rank
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

            # generates grid spatial coordinates
            locations = generate_grid_coordinates(img.spatial_rank,
                                                  img.data.shape,
                                                  self.patch.image_size,
                                                  self.grid_size)
            n_patches = locations.shape[0]
            extra_patches = batch_size - n_patches % batch_size \
                if (n_patches % batch_size) != 0 else 0
            extend_n_patches = n_patches + extra_patches
            if extra_patches > 0:
                print("yielding {} locations, "
                      "extends to {} to be divisible by batch size {}".format(
                    n_patches, extend_n_patches, batch_size))
            else:
                print("yielding {} locations".format(n_patches))

            for i in range(0, extend_n_patches):
                loc = locations[i % n_patches]
                self.patch.set_data(idx, loc, img, seg, weight_map)
                yield self.patch
