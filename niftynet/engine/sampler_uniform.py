# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from copy import deepcopy

import numpy as np
import tensorflow as tf

import niftynet.utilities.misc_io as io
from niftynet.engine.base_sampler import BaseSampler
from niftynet.layer.base_layer import Layer
from niftynet.engine.input_buffer import InputBatchQueueRunner


def rand_spatial_coordinates(img_size, win_size, n_samples):
    # Please see grid_sampler.py generate_grid_coordinates() for more info
    if np.any([d < win_size for d in img_size[:len(win_size)]]):
        raise ValueError('Window size larger than the input image dims'
                         ' (please make sure that image dims after '
                         'random spatial scaling are still larger than window'
                         ' size)')
    # consisting of starting and ending coordinates
    all_coords = np.zeros((n_samples, int(spatial_rank * 2.0)), dtype=np.int)

    for i in range(0, grid_spatial_rank):
        all_coords[:, i] = np.random.randint(
            0, max(img_size[i] - win_size, 1), n_samples)
        all_coords[:, i + full_spatial_rank] = all_coords[:, i] + win_size
    if spatial_rank == 2.5:
        all_coords[:, 2] = np.random.randint(
            0, max(img_size[2] - 1, 1), n_samples)
    return all_coords


class UniformSampler(Layer, InputBatchQueueRunner):
    """
    This class generators samples by uniformly sampling each input volume
    currently 4D input is supported, Height x Width x Depth x Modality
    """

    def __init__(self, reader, data_param):
        self.reader = reader
        self.window_sizes = {field: self.infer_window_sizes(field,
                                                            data_param)
                             for field in self.reader.output_fields}
        self.placeholders = {field: self.create_placeholders(field)
                             for field in self.reader.output_fields}

        tf.logging.info(
            "initialising sampler output {}".format(self.window_sizes))
        InputBatchQueueRunner.__init__(self, 3, 10, self.placeholders)
        import pdb; pdb.set_trace()

    def create_placeholders(self, field):
        dtype = set(self.reader.dtypes[field]).pop()
        return tf.placeholder(dtype=dtype, name=field)

    def infer_window_sizes(self, field_name, input_data_param):
        # read window_size property and group them based on output_fields
        inputs = self.reader.input_sources[field_name]
        window_sizes = [input_data_param[input_name].window_size
                        for input_name in inputs]
        if not all(window_sizes):
            window_sizes = filter(None, window_sizes)
        uniq_window = set(window_sizes)
        if len(uniq_window) > 1:
            raise NotImplementedError(
                "trying to combine input sources "
                "with different window sizes: {}".format(window_sizes))
        if not uniq_window:
            raise ValueError(
                "window_size undetermined{}".format(self.reader.input_sources))
        uniq_window = list(uniq_window.pop())
        while len(uniq_window) < 4:
            # expanding the window size, so that we can
            # concatenate on axis=3 the modality dim
            uniq_window.append(1)
        uniq_window[3] = uniq_window[3] * len(inputs)
        # integer window sizes supported
        uniq_window = tuple(map(int, uniq_window))
        return uniq_window

    def layer_op(self):
        return
    # def __init__(self,
    #             patch,
    #             volume_loader,
    #             patch_per_volume=1,
    #             data_augmentation_methods=None,
    #             name="uniform_sampler"):

    #    super(UniformSampler, self).__init__(patch=patch, name=name)
    #    self.volume_loader = volume_loader
    #    self.patch_per_volume = patch_per_volume
    #    if data_augmentation_methods is None:
    #        self.data_augmentation_layers = []
    #    else:
    #        self.data_augmentation_layers = data_augmentation_methods

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

            # generates random spatial coordinates
            locations = rand_spatial_coordinates(img.spatial_rank,
                                                 img.data.shape,
                                                 patch.image_size,
                                                 self.patch_per_volume)
            for loc in locations:
                patch.set_data(idx, loc, img, seg, weight_map)
                yield patch
