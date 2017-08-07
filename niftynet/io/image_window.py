# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

N_SPATIAL = 3
LOCATION_FORMAT = "{}_location"
BUFFER_POSITION_DTYPE = tf.int32
TF_NP_DTYPES = {tf.int32: np.int32, tf.float32: np.float32}


class ImageWindow(object):
    def __init__(self, fields, shapes, dtypes):
        self.fields = fields
        self.shapes = shapes
        self.dtypes = dtypes

        self.n_samples = None
        self._placeholders_dict = None

    @classmethod
    def from_user_spec(cls,
                       source_names,
                       image_shapes,
                       image_dtypes,
                       data_param):
        input_fields = tuple(source_names)
        # complete window shapes based on user input and input_image sizes
        spatial_shapes = {
            name: read_window_sizes(source_names[name], data_param)
            for name in input_fields}
        shapes = {
            name: complete_partial_window_sizes(
                spatial_shapes[name], image_shapes[name])
            for name in input_fields}
        # create ImageWindow instance
        return cls(input_fields, shapes, image_dtypes)

    def placeholders_dict(self, n_samples=1):
        if self._placeholders_dict is None or self.n_samples is None:
            # create placeholders, required as network input
            names = list(self.fields)
            placeholders = [
                tf.placeholder(dtype=self.dtypes[name],
                               shape=[n_samples] + list(self.shapes[name]))
                for name in self.fields]

            # extending names placeholders with fields of coordinates
            names.extend([LOCATION_FORMAT.format(name) for name in names])
            placeholders.extend(
                [tf.placeholder(dtype=BUFFER_POSITION_DTYPE,
                                shape=(n_samples, 1 + N_SPATIAL * 2))
                 for _ in self.fields])

            self.n_samples = n_samples
            self._placeholders_dict = dict(zip(names, placeholders))
        return self._placeholders_dict

    def data_dict(self):
        # dictionary required by tf queue, {placeholder:data_array}
        output_dict = {}
        for name, placeholder in self._placeholders_dict.items():
            shape = placeholder.shape.as_list()
            np_dtype = TF_NP_DTYPES.get(placeholder.dtype, np.float32)
            output_dict[placeholder] = np.zeros(shape, dtype=np_dtype)
        return output_dict

    def coordinates_placeholder(self, field):
        return self._placeholders_dict[LOCATION_FORMAT.format(field)]

    def image_data_placeholder(self, field):
        return self._placeholders_dict[field]


def read_window_sizes(input_mod_list, input_data_param):
    # read window_size property and group them based on output_fields
    window_sizes = [input_data_param[input_name].spatial_window_size
                    for input_name in input_mod_list]
    if not all(window_sizes):
        window_sizes = filter(None, window_sizes)
    uniq_window = set(window_sizes)
    if len(uniq_window) > 1:
        raise NotImplementedError(
            "trying to combine input sources "
            "with different window sizes: {}".format(window_sizes))
    if not uniq_window:
        raise ValueError(
            "window_size undetermined{}".format(input_mod_list))
    uniq_window = list(uniq_window.pop())
    # integer window sizes supported
    uniq_window = tuple(map(int, uniq_window))
    return uniq_window


def complete_partial_window_sizes(win_size, img_size):
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
