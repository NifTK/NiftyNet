# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import os

import numpy as np
import tensorflow as tf

import niftynet.io.misc_io as misc_io
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.pad import PadLayer


class ImageWindowsAggregator(object):
    def __init__(self, image_reader=None):
        self.reader = image_reader
        self._image_id = None

    @property
    def input_image(self):
        if self.image_id is not None and self.reader:
            return self.reader.output_list[self.image_id]
        else:
            return None

    @property
    def image_id(self):
        return self._image_id

    @image_id.setter
    def image_id(self, current_id):
        try:
            self._image_id = int(current_id)
        except ValueError:
            tf.logging.fatal("unknown image id format (should be an integer")

    def decode_batch(self, *args, **kwargs):
        raise NotImplementedError


class GridSamplesAggregator(ImageWindowsAggregator):
    def __init__(self,
                 image_reader,
                 output_path='./',
                 window_border=(),
                 interp_order=0):
        ImageWindowsAggregator.__init__(self, image_reader=image_reader)
        self.image_out = None
        self.output_path = os.path.abspath(output_path)
        self.window_border = window_border
        self.output_interp_order = interp_order

    def decode_batch(self, window, location):
        n_samples = location.shape[0]
        window, location = crop_batch(window, location, self.window_border)

        for batch_id in range(n_samples):
            image_id, x_, y_, z_, _x, _y, _z = location[batch_id, :]
            if image_id != self.image_id:
                self._save_current_image()
                if self._is_stopping_signal(location[batch_id]):
                    return False
                self.image_out = self._initialise_empty_image(
                    image_id=image_id,
                    n_channels=window.shape[-1],
                    dtype=window.dtype)
            self.image_out[x_:_x, y_:_y, z_:_z, ...] = window[batch_id, ...]
        return True

    def _initialise_empty_image(self, image_id, n_channels, dtype=np.float):
        self.image_id = image_id
        spatial_shape = self.input_image['image'].shape[:3]
        output_image_shape = spatial_shape + (n_channels,)
        empty_image = np.zeros(output_image_shape, dtype=dtype)

        for layer in self.reader.preprocessors:
            if isinstance(layer, PadLayer):
                empty_image, _ = layer(empty_image)
        return empty_image

    def _save_current_image(self):
        if self.input_image is None:
            return

        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, PadLayer):
                self.image_out, _ = layer.inverse_op(self.image_out)
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                self.image_out, _ = layer.inverse_op(self.image_out)
        subject_name = self.reader.get_subject_id(self.image_id)
        filename = "{}_niftynet_out.nii.gz".format(subject_name)
        source_image_obj = self.input_image['image']
        misc_io.save_data_array(self.output_path,
                                filename,
                                self.image_out,
                                source_image_obj,
                                self.output_interp_order)
        return

    @staticmethod
    def _is_stopping_signal(location_vector):
        return np.all(location_vector[1:4] + location_vector[4:7]) == 0


def crop_batch(window, location, border):
    if border == ():
        return window, location
    assert len(border) == 3, "unknown border format (should be an array of" \
                             "three elements corresponding to 3 spatial dims"

    window_shape = window.shape
    if len(window_shape) != 5:
        raise NotImplementedError(
            "window shape not supported: {}".format(window_shape))
    spatial_shape = window_shape[1:4]
    assert all([win_size > 2 * border_size
                for (win_size, border_size) in zip(spatial_shape, border)]), \
        "window sizes should be larger than inference border size * 2"
    window = window[:,
             border[0]:spatial_shape[0] - border[0],
             border[1]:spatial_shape[1] - border[1],
             border[2]:spatial_shape[2] - border[2], ...]
    for idx in range(3):
        location[:, idx + 1] = location[:, idx + 1] + border[idx]
        location[:, idx + 4] = location[:, idx + 4] - border[idx]
    return window, location
