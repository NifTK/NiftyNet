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
    currently 4D input is supported, Hight x Width x Depth x Modality
    """

    def __init__(self,
                 patch,
                 volume_loader,
                 patch_per_volume=1,
                 volume_padding_size=0,
                 name="uniform_sampler"):

        super(UniformSampler, self).__init__(patch=patch, name=name)
        self.volume_loader = volume_loader
        self.patch_per_volume = patch_per_volume
        self.volume_padding_size = volume_padding_size

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
            img = io.match_volume_shape_to_patch_definition(
                img, self.patch.full_image_shape)
            seg = io.match_volume_shape_to_patch_definition(
                seg, self.patch.full_label_shape)
            weight_map = io.match_volume_shape_to_patch_definition(
                weight_map, self.patch.full_weight_map_shape)
            if img.ndim - spatial_rank > 1:
                raise NotImplementedError
                # time series data are not supported after this point

            if self.volume_padding_size > 0:
                img = io.volume_spatial_padding(
                    img, self.volume_padding_size)
                seg = io.volume_spatial_padding(
                    seg, self.volume_padding_size)
                weight_map = io.volume_spatial_padding(
                    weight_map, self.volume_padding_size)

            # generates random spatial coordinates
            location = rand_spatial_coordinates(spatial_rank,
                                                img.shape,
                                                self.patch.image_size,
                                                self.patch_per_volume)

            for t in range(0, self.patch_per_volume):

                self.patch.info = np.array(np.hstack([[idx], location[t]]),
                                           dtype=np.int64)
                if spatial_rank == 3:
                    x_, y_, z_, _x, _y, _z = location[t]
                    self.patch.image = img[x_:_x, y_:_y, z_:_z, :]
                    if self.patch.has_labels:
                        diff = self.patch.image_size - self.patch.label_size
                        assert diff >= 0  # assumes label_size <= image_size
                        x_d, y_d, z_d = (x_ + diff), (y_ + diff), (z_ + diff)
                        self.patch.label = \
                            seg[x_d: (self.patch.label_size + x_d),
                                y_d: (self.patch.label_size + y_d),
                                z_d: (self.patch.label_size + z_d),
                                :]

                    if self.patch.has_weight_maps:
                        diff = self.patch.image_size - self.patch.weight_map_size
                        assert diff >= 0
                        x_d, y_d, z_d = (x_ + diff), (y_ + diff), (z_ + diff)
                        self.patch.weight_map = \
                            weight_map[x_d: (self.patch.weight_map_size + x_d),
                                       y_d: (self.patch.weight_map_size + y_d),
                                       z_d: (self.patch.weight_map_size + z_d),
                                       :]
                    yield self.patch

                elif spatial_rank == 2:
                    x_, y_, _x, _y, = location[t]
                    self.patch.image = img[x_:_x, y_:_y, :]
                    if self.patch.has_labels:
                        diff = self.patch.image_size - self.patch.label_size
                        assert diff >= 0  # assumes label_size <= image_size
                        x_d, y_d = (x_ + diff), (y_ + diff)
                        self.patch.label = \
                            seg[x_d: (self.patch.label_size + x_d),
                                y_d: (self.patch.label_size + y_d),
                                :]

                    if self.patch.has_weight_maps:
                        diff = self.patch.image_size - self.patch.weight_map_size
                        assert diff >= 0
                        x_d, y_d, = (x_ + diff), (y_ + diff)
                        self.patch.weight_map = \
                            weight_map[x_d: (self.patch.weight_map_size + x_d),
                                       y_d: (self.patch.weight_map_size + y_d),
                                       :]
                    yield self.patch
