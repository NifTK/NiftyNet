# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from copy import deepcopy

import sys
import numpy as np
import tensorflow as tf

import niftynet.utilities.misc_io as io
from niftynet.engine.base_sampler import BaseSampler
from niftynet.layer.base_layer import Layer
from niftynet.engine.input_buffer import InputBatchQueueRunner


BUFFER_IMAGE_DTYPE = tf.float32
BUFFER_POSITION_DTYPE = tf.uint16


def _complete_partial_window_sizes(win_size, img_size):
    img_ndims = len(img_size)
    # crop win_size list if it's longer than img_size
    win_size = win_size[:img_ndims]
    # complete win_size list if it's shorter than img_size
    while len(win_size) < img_ndims:
        win_size.append(img_size[len(win_size)])
    # replace zero with full length in the n-th dim of image
    win_size = [win if win > 0 else sys.maxint for win in win_size]
    win_size = [min(win, img) for(win, img) in zip(win_size, img_size)]
    return win_size


def rand_spatial_coordinates(img_sizes, win_sizes, n_samples):
    if len(set([img_size[:3] for img_size in img_sizes])) > 1:
        tf.logging.fatal("Don't know how to generate sampling "
                         "locations:Spatial dimensions of the "
                         "grouped input sources are not "
                         "consistent. {}".format(img_sizes))
        raise NotImplementedError
    all_coords = []
    for mod_id, img_size in enumerate(img_sizes):
        img_ndims = len(img_size)
        win_size = list(win_sizes[mod_id])
        win_size = _complete_partial_window_sizes(win_size, img_size)
        win_sizes[mod_id] = win_size

    import pdb; pdb.set_trace()
    coords = np.zeros((n_samples, len(img_size)*2), dtype=np.uint16)
    for i in range(0, grid_spatial_rank):
        coords[:, i] = np.random.randint(
            0, max(img_size[i] - win_size, 1), n_samples)
        coords[:, i + full_spatial_rank] = all_coords[:, i] + win_size
    all_coords.append(coords)
    return all_coords


class UniformSampler(Layer, InputBatchQueueRunner):
    """
    This class generators samples by uniformly sampling each input volume
    currently 4D input is supported, Height x Width x Depth x Modality
    """

    def __init__(self, reader, data_param, samples_per_subject):
        self.reader = reader

        # create window sizes and placeholders
        self.window_sizes = {field: self.infer_window_sizes(field, data_param)
                             for field in self.reader.output_fields}
        self.placeholders = {field: self.create_placeholders(field)
                             for field in self.reader.output_fields}
        self.placeholders['position'] = tf.placeholder(
            dtype=BUFFER_POSITION_DTYPE, name='position')
        tf.logging.info(
            "initialised sampler output {}".format(self.window_sizes))

        # initialise input buffer
        InputBatchQueueRunner.__init__(self, 3, 10, self.placeholders)

        # find spatial location
        image_sizes = [image.get_data().shape # TODO: padding
                       for image in list(self.reader().values())]
        window_sizes = list(self.window_sizes.values())
        coords = rand_spatial_coordinates(image_sizes,
                                          window_sizes,
                                          samples_per_subject)

    def create_placeholders(self, field):
        # TODO: decide data types from the inputs??
        #dtype = set(self.reader.dtypes[field]).pop()
        #return tf.placeholder(dtype=dtype, name=field)
        return tf.placeholder(dtype=BUFFER_IMAGE_DTYPE, name=field)

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
