# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from copy import deepcopy

import sys
import numpy as np
import tensorflow as tf

import niftynet.utilities.misc_io as io
from niftynet.engine.base_sampler import BaseSampler
from niftynet.layer.base_layer import Layer
from niftynet.engine.input_buffer import InputBatchQueueRunner


BUFFER_IMAGE_DTYPE = tf.float32
BUFFER_POSITION_DTYPE = tf.int32
N_SPATIAL = 3
LOCATION_FORMAT = "{}_location"


def _complete_partial_window_sizes(win_size, img_size):
    img_ndims = len(img_size)
    # crop win_size list if it's longer than img_size
    win_size = list(win_size[:img_ndims])
    # complete win_size list if it's shorter than img_size
    while len(win_size) < img_ndims:
        win_size.append(img_size[len(win_size)])
    # replace zero with full length in the n-th dim of image
    win_size = [win if win > 0 else sys.maxint for win in win_size]
    win_size = [min(win, img) for(win, img) in zip(win_size, img_size)]
    return win_size



class UniformSampler(Layer, InputBatchQueueRunner):
    """
    This class generators samples by uniformly sampling each input volume
    currently 4D input is supported, Height x Width x Depth x Modality
    """

    def __init__(self, reader, data_param, samples_per_subject):
        self.reader = reader
        self.samples_per_subject = samples_per_subject
        # create window sizes and placeholders
        self.window_sizes = {field: self.infer_window_sizes(field, data_param)
                             for field in self.reader.output_fields}
        data = self.reader()
        self.img_sizes = {field: image.get_data().shape # TODO: padding
                          for (field, image) in data.items()}
        # complete user's input by matching the image dims
        for mod_id, img_size in self.img_sizes.items():
            win_size = _complete_partial_window_sizes(
                self.window_sizes[mod_id], img_size)
            self.window_sizes[mod_id] = win_size

        # find spatial location
        self.placeholders = self.create_placeholders(data_param)
        # initialise input buffer
        InputBatchQueueRunner.__init__(self, 3, 10, self.placeholders)
        Layer.__init__(self, name='input_buffer')
        tf.logging.info(
            "initialised sampler output {}".format(self.window_sizes))
        print(self.reader.output_list[0]['image'].get_data().shape)
        print(self.reader.output_list[0]['label'].get_data().shape)
        #for i in self():
        #    print(i)
        #self.reader()['image'].get_data(param['image'])
        import pdb; pdb.set_trace()

    def create_placeholders(self, data_param):
        # TODO: decide data types from the inputs??
        #dtype = set(self.reader.dtypes[field]).pop()
        #return tf.placeholder(dtype=dtype, name=field)
        placeholders = {field: tf.placeholder(dtype=BUFFER_IMAGE_DTYPE,
                                              shape=self.window_sizes[field],
                                              name=field)
                             for field in self.reader.output_fields}
        for field in list(placeholders):
            placeholders[LOCATION_FORMAT.format(field)] = tf.placeholder(
                dtype=BUFFER_POSITION_DTYPE,
                shape=(1+N_SPATIAL*2),
                name=LOCATION_FORMAT.format(field))
        return placeholders

    def infer_window_sizes(self, field_name, input_data_param):
        # read window_size property and group them based on output_fields
        inputs = self.reader.input_sources[field_name]
        window_sizes = [input_data_param[input_name].spatial_window_size
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
        while True:
            data = self.reader()
            image_sizes = {field: image.get_data().shape # TODO: padding
                           for (field, image) in data.items()}
            if not data:
                break;
            coordinates = self.rand_spatial_coordinates(self.reader.current_id, image_sizes)
            for i in range(self.samples_per_subject):
                output_dict = {}
                for image_name in list(data):
                    coords = coordinates[LOCATION_FORMAT.format(image_name)]
                    coord = coords[i, :]
                    x_, y_, z_, _x, _y, _z = coord[1:]
                    image = data[image_name].get_data().astype(np.float32)
                    window = image[x_:_x, y_:_y, z_:_z,...]
                    output_dict[self.placeholders[image_name]] = window
                    output_dict[self.placeholders[LOCATION_FORMAT.format(image_name)]] = coord
                import pdb; pdb.set_trace()
            yield output_dict

    def rand_spatial_coordinates(self, subject_id, img_sizes):
        win_sizes = self.window_sizes
        n_samples = self.samples_per_subject
        if len(set([img_size[:3] for img_size in list(img_sizes.values())])) > 1:
            tf.logging.fatal("Don't know how to generate sampling "
                             "locations: Spatial dimensions of the "
                             "grouped input sources are not "
                             "consistent. {}".format(img_sizes))
            raise NotImplementedError
        # complete user's input by matching the image dims
        for mod_id, img_size in img_sizes.items():
            win_size = _complete_partial_window_sizes(win_sizes[mod_id], img_size)
            win_sizes[mod_id] = win_size[:N_SPATIAL]

        # find spatial window location based on the largest spatial window
        win_sizes_array = np.asarray(win_sizes.values(), dtype=np.int32)
        max_spatial_win = np.max(win_sizes_array, axis=0)
        max_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
        # first image for the image shape
        img_spatial_size = img_sizes.values()[0][:N_SPATIAL]
        for i in range(0, N_SPATIAL):
            max_coords[:, i] = np.random.randint(
                0, max(img_spatial_size[i] - max_spatial_win[i], 1), n_samples)

        # adjust max spatial coordinates based on each spatial window size
        all_coordinates = {}
        for (mod_id, win_size) in win_sizes.items():
            subject_id = np.ones((n_samples,), dtype=np.int32) * subject_id
            half_win_diff = np.floor((max_spatial_win - win_sizes[mod_id]) / 2.0)
            spatial_coords = np.zeros((n_samples, N_SPATIAL*2), dtype=np.int32)
            spatial_coords[:, :N_SPATIAL] = \
                    max_coords[:, :N_SPATIAL] + half_win_diff[:N_SPATIAL]
            spatial_coords[:, N_SPATIAL:] = \
                    spatial_coords[:, :N_SPATIAL] + win_size[:N_SPATIAL]
            spatial_coords = np.append(subject_id[:,None], spatial_coords, axis=1)
            all_coordinates[LOCATION_FORMAT.format(mod_id)] = spatial_coords
        return all_coordinates

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

    #def layer_op(self, batch_size=1):
    #    """
    #     problems:
    #        check how many modalities available
    #        check the colon operator
    #        automatically handle mutlimodal by matching dims?
    #    """
    #    spatial_rank = self.patch.spatial_rank
    #    local_layers = [deepcopy(x) for x in self.data_augmentation_layers]
    #    patch = deepcopy(self.patch)
    #    while self.volume_loader.has_next:
    #        img, seg, weight_map, idx = self.volume_loader()

    #        # to make sure all volumetric data have the same spatial dims
    #        # and match volumetric data shapes to the patch definition
    #        # (the matched result will be either 3d or 4d)
    #        img.spatial_rank = spatial_rank

    #        img.data = io.match_volume_shape_to_patch_definition(
    #            img.data, patch)
    #        if img.data.ndim == 5:
    #            raise NotImplementedError
    #            # time series data are not supported
    #        if seg is not None:
    #            seg.spatial_rank = spatial_rank
    #            seg.data = io.match_volume_shape_to_patch_definition(
    #                seg.data, patch)
    #        if weight_map is not None:
    #            weight_map.spatial_rank = spatial_rank
    #            weight_map.data = io.match_volume_shape_to_patch_definition(
    #                weight_map.data, patch)

    #        # apply volume level augmentation
    #        for aug in local_layers:
    #            aug.randomise(spatial_rank=spatial_rank)
    #            img, seg, weight_map = aug(img), aug(seg), aug(weight_map)

    #        # generates random spatial coordinates
    #        locations = rand_spatial_coordinates(img.spatial_rank,
    #                                             img.data.shape,
    #                                             patch.image_size,
    #                                             self.patch_per_volume)
    #        for loc in locations:
    #            patch.set_data(idx, loc, img, seg, weight_map)
    #            yield patch
