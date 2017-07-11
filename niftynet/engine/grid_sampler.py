# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np

import niftynet.utilities.misc_io as io
from niftynet.engine.base_sampler import BaseSampler

SUPPORTED_SPATIAL_RANKS = {2.0, 2.5, 3.0}


def generate_grid_coordinates(spatial_rank, img_size, win_size, grid_size):
    """
    Generate N-D coordinates with a fixed step size 'grid_size' in each dim

    For 2.5D volume (3D volume treated as a stack of 2D slices),
    given the volume input is [HWDC],
    the grid should be constructed on [HW] and running through all possible 'D's

    :param spatial_rank: the number of spatial dims
    :param img_size: image size to be covered by the sampling grid
    :param win_size: window size centered at each sampling point
    :param grid_size: step size of the samples
    :return: n*2 columns of coordinates for n-d image size
    """
    assert spatial_rank in SUPPORTED_SPATIAL_RANKS
    if grid_size <= 0:
        return None
    grid_spatial_rank = int(np.floor(spatial_rank))
    full_spatial_rank = int(np.ceil(spatial_rank))
    assert np.all([d >= win_size for d in img_size[:grid_spatial_rank]])

    # generating sampling points along each dim
    steps_along_each = [_enumerate_step_points(0,
                                               img_size[i],
                                               win_size,
                                               grid_size)
                        for i in range(0, grid_spatial_rank)]
    if spatial_rank == 2.5:
        # the third dimension should be sampled densely
        steps_along_each.append(_enumerate_step_points(0, img_size[2], 1, 1))
    # create a mesh grid
    starting_ = np.asarray(np.meshgrid(*steps_along_each))
    starting_ = starting_.reshape((full_spatial_rank, -1))
    # transform mesh grid into a list of coordinates
    all_coordinates = np.zeros((starting_.shape[1], int(spatial_rank*2.0)),
                               dtype=np.int)
    for i in range(0, grid_spatial_rank):
        all_coordinates[:, i] = starting_[i, :]
        all_coordinates[:, i + full_spatial_rank] = starting_[i, :] + win_size
    if spatial_rank == 2.5:
        all_coordinates[:, 2] = starting_[2, :]
    # the order of the coordinates should be consistent with
    # input_placeholders.py : full_info_shape
    # so that we have a consistent interpretation
    return all_coordinates


def _enumerate_step_points(starting, ending, win_size, step_size):
    """
    generate all possible sampling size in between starting and ending
    :param starting: integer of starting value
    :param ending: integer of ending value
    :param win_size: integer of window length
    :param step_size: integer of distance between two sampling points
    :return: a set of unique sampling points
    """
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
                img.data, self.patch)
            if img.data.ndim == 5:
                raise NotImplementedError
                # time series data are not supported
            if seg is not None:
                seg.spatial_rank = spatial_rank
                seg.data = io.match_volume_shape_to_patch_definition(
                    seg.data, self.patch)
            if weight_map is not None:
                weight_map.spatial_rank = spatial_rank
                weight_map.data = io.match_volume_shape_to_patch_definition(
                    weight_map.data,
                    self.patch)

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
