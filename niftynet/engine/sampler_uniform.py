# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

from niftynet.engine.input_buffer import InputBatchQueueRunner
from niftynet.layer.base_layer import Layer

BUFFER_POSITION_DTYPE = tf.int32
N_SPATIAL = 3
LOCATION_FORMAT = "{}_location"
NP_TF_DTYPES = {'i': tf.int32,
                'u': tf.int32,
                'b': tf.int32,
                'f': tf.float32}

TF_NP_DTYPES = {tf.int32: np.int32,
                tf.float32: np.float32}


def _complete_partial_window_sizes(win_size, img_size):
    img_ndims = len(img_size)
    # crop win_size list if it's longer than img_size
    win_size = list(win_size[:img_ndims])
    # complete win_size list if it's shorter than img_size
    while len(win_size) < img_ndims:
        win_size.append(img_size[len(win_size)])
    # replace zero with full length in the n-th dim of image
    win_size = [win if win > 0 else sys.maxint for win in win_size]
    win_size = [min(win, img) for (win, img) in zip(win_size, img_size)]
    return tuple(win_size)


def infer_tf_dtypes(image_object):
    uniq_np_dtype = set(image_object.dtype)
    if len(uniq_np_dtype) > 1:
        # heterogeneous input data types, promoting to floatings
        return NP_TF_DTYPES.get('f', None)
    else:
        return NP_TF_DTYPES.get(uniq_np_dtype.pop().kind, None)


class UniformSampler(Layer, InputBatchQueueRunner):
    """
    This class generators samples by uniformly sampling each input volume
    currently 4D input is supported, Height x Width x Depth x Modality
    """

    def __init__(self, reader, data_param, samples_per_subject):
        self.reader = reader
        self.samples_per_subject = samples_per_subject
        # initialise input buffer
        InputBatchQueueRunner.__init__(self, 3, 10, True)
        Layer.__init__(self, name='input_buffer')

        names = list(self.reader.output_fields)
        first_input = self.reader.output_list[0]
        # TODO: padding
        input_sizes = [first_input[name].get_data().shape for name in names]

        # create image placeholders
        dtypes = [infer_tf_dtypes(first_input[name]) for name in names]
        spatial_shapes = [self.read_window_sizes(name, data_param)
                          for name in names]
        self.shapes = [_complete_partial_window_sizes(win_size, img_size)
                       for win_size, img_size in
                       zip(spatial_shapes, input_sizes)]
        placeholders = [
            tf.placeholder(
                dtype=dtype,
                shape=[self.samples_per_subject] + list(shape))
            for (dtype, shape) in zip(dtypes, self.shapes)]
        # extend with location placeholders
        names.extend([LOCATION_FORMAT.format(name) for name in names])
        placeholders.extend(
            [tf.placeholder(
                dtype=BUFFER_POSITION_DTYPE,
                shape=(self.samples_per_subject, 1 + N_SPATIAL * 2))
                for _ in names])
        self.placeholders_dict = dict(zip(names, placeholders))
        self._create_queue_and_ops(self.placeholders_dict)

        # # create window sizes and placeholders
        # self.window_sizes = {field: self.read_window_sizes(field, data_param)
        #                      for field in self.reader.output_fields}
        # data = self.reader()
        # self.img_sizes = {field: image.get_data().shape
        #                   for (field, image) in data.items()}
        # # complete user's input by matching the image dims
        # for mod_id, img_size in self.img_sizes.items():
        #     win_size = _complete_partial_window_sizes(
        #         self.window_sizes[mod_id], img_size)
        #     self.window_sizes[mod_id] = win_size
        tf.logging.info("initialised sampler output {}".format(self.shapes))
        for output_dict in self():
            print(output_dict)

    def read_window_sizes(self, field_name, input_data_param):
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
            image_sizes = [data[name].get_data().shape
                           for name in self.reader.output_fields]

            if not data:
                break
            coordinates = rand_spatial_coordinates(self.reader.current_id,
                                                   self.shapes,
                                                   image_sizes,
                                                   self.samples_per_subject)
            coordinates = dict(zip(self.reader.output_fields, coordinates))
            #  initialise output dict
            output_dict = {}
            for name, placeholder in self.placeholders_dict.items():
                shape = placeholder.shape.as_list()
                dtype = TF_NP_DTYPES.get(placeholder.dtype, np.float32)
                output_dict[placeholder] = np.zeros(shape, dtype=dtype)

            for name in list(data):
                location_array = output_dict[
                    self.placeholders_dict[LOCATION_FORMAT.format(name)]]
                location_array[...] = coordinates[name]

                image = data[name].get_data()
                image_array = output_dict[self.placeholders_dict[name]]
                for (i, location) in enumerate(location_array[:, 1:]):
                    x_, y_, z_, _x, _y, _z = location
                    image_array[i, ...] = image[x_:_x, y_:_y, z_:_z, ...]
            yield output_dict


def rand_spatial_coordinates(subject_id, win_sizes, img_sizes, n_samples):
    if len(set(img_size[:N_SPATIAL] for img_size in img_sizes)) > 1:
        tf.logging.fatal("Don't know how to generate sampling "
                         "locations: Spatial dimensions of the "
                         "grouped input sources are not "
                         "consistent. {}".format(img_sizes))
        raise NotImplementedError

    # find spatial window location based on the largest spatial window
    spatial_win_sizes = [win_size[:N_SPATIAL] for win_size in win_sizes]
    spatial_win_sizes = np.asarray(spatial_win_sizes, dtype=np.int32)
    max_spatial_win = np.max(spatial_win_sizes, axis=0)
    max_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    for i in range(0, N_SPATIAL):
        max_coords[:, i] = np.random.randint(
            0, max(img_sizes[0][i] - max_spatial_win[i], 1), n_samples)

    # adjust max spatial coordinates based on each spatial window size
    all_coordinates = []
    for win_size in spatial_win_sizes:
        subject_id = np.ones((n_samples,), dtype=np.int32) * subject_id
        spatial_coords = np.zeros(
            (n_samples, N_SPATIAL * 2), dtype=np.int32)
        # shift starting coords of the window
        # so that smaller windows are centred within the large windows
        half_win_diff = np.floor((max_spatial_win - win_size) / 2.0)
        spatial_coords[:, :N_SPATIAL] = \
            max_coords[:, :N_SPATIAL] + half_win_diff[:N_SPATIAL]

        spatial_coords[:, N_SPATIAL:] = \
            spatial_coords[:, :N_SPATIAL] + win_size[:N_SPATIAL]
        # include the subject id
        spatial_coords = np.append(
            subject_id[:, None], spatial_coords, axis=1)
        all_coordinates.append(spatial_coords)
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

    # def layer_op(self, batch_size=1):
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
