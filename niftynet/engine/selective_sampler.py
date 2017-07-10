# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np

import niftynet.utilities.misc_io as io
from niftynet.engine.base_sampler import BaseSampler
from niftynet.engine.uniform_sampler import rand_spatial_coordinates


class SelectiveSampler(BaseSampler):
    """
    This class generators samples by sampling each input volume
    the output samples satisfy constraints such as number of
    unique values in training label
    (currently 4D input is supported, Height x Width x Depth x Modality)
    """

    def __init__(self,
                 patch,
                 volume_loader,
                 spatial_location_check=None,
                 data_augmentation_methods=None,
                 patch_per_volume=1,
                 name="selective_sampler"):

        super(SelectiveSampler, self).__init__(patch=patch, name=name)
        self.volume_loader = volume_loader
        self.patch_per_volume = patch_per_volume
        self.spatial_location_check = spatial_location_check
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
        # batch_size is needed here so that it generates total number of
        # N samples where (N % batch_size) == 0

        spatial_rank = self.patch.spatial_rank
        local_layers = [deepcopy(x) for x in self.data_augmentation_layers]
        patch = deepcopy(self.patch)
        spatial_location_check = deepcopy(self.spatial_location_check)
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

            # apply volume level augmentation
            for aug in local_layers:
                aug.randomise(spatial_rank=spatial_rank)
                img, seg, weight_map = aug(img), aug(seg), aug(weight_map)

            # to generate 'patch_per_volume' samples satisfying
            #  the conditions specified by spatial_location_check instance:
            # 'n_locations_to_check' samples are randomly sampled
            # and checked. This sampling and checking process is repeated
            # for 10 times at most.
            locations = []
            n_locations_to_check = self.patch_per_volume * 10
            spatial_location_check.sampling_from(seg.data)
            n_trials = 10000
            while len(locations) < self.patch_per_volume and n_trials > 0:
                # generates random spatial coordinates
                candidate_locations = rand_spatial_coordinates(
                    img.spatial_rank,
                    img.data.shape,
                    patch.image_size,
                    1)
                is_valid = [spatial_location_check(location, spatial_rank)
                            for location in candidate_locations]
                is_valid = np.asarray(is_valid, dtype=bool)
                # print("{} good samples from {} candidates".format(
                #     np.sum(is_valid), len(candidate_locations)))
                for loc in candidate_locations[is_valid]:
                    locations.append(loc)
                n_trials -= 1

            while len(locations) < self.patch_per_volume:
                candidate_locations = rand_spatial_coordinates(
                    img.spatial_rank,
                    img.data.shape,
                    patch.image_size,
                    1)
                for loc in candidate_locations:
                    locations.append(loc)
            locations = np.vstack(locations)
            print(len(locations))
            for loc in locations:
                patch.set_data(idx, loc, img, seg, weight_map)
                yield patch
