# -*- coding: utf-8 -*-
"""
windows aggregator decode sampling grid coordinates and image id from
batch data, forms image level output and write to hard drive.
"""
from __future__ import absolute_import, print_function, division

import os

import numpy as np

from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator
from niftynet.layer.pad import PadLayer


class GridSamplesAggregator(ImageWindowsAggregator):
    """
    This class keeps record of the currently cached image,
    initialised as all zeros, and the values are replaced
    by image window data decoded from batch.
    """
    def __init__(self,
                 image_reader,
                 image_writer,
                 name='image',
                 window_border=(),
                 fill_constant=0.0):
        ImageWindowsAggregator.__init__(
            self,
            image_reader,
            image_writer)

        self.name = name
        self.image_out = None
        self.window_border = window_border
        self.fill_constant = fill_constant

    def decode_batch(self, window, location):
        n_samples = location.shape[0]
        window, location = self.crop_batch(window, location, self.window_border)

        for batch_id in range(n_samples):
            image_id, x_start, y_start, z_start, x_end, y_end, z_end = \
                location[batch_id, :]
            if image_id != self.image_id:
                # image name changed:
                #    save current image and create an empty image
                self._save_current_image()
                if self._is_stopping_signal(location[batch_id]):
                    return False
                self.image_out = self._initialise_empty_image(
                    image_id=image_id,
                    n_channels=window.shape[-1],
                    dtype=window.dtype)
            self.image_out[x_start:x_end,
                           y_start:y_end,
                           z_start:z_end, ...] = window[batch_id, ...]
        return True

    def _initialise_empty_image(self, image_id, n_channels, dtype=np.float):
        self.image_id = image_id
        spatial_shape = self.input_image[self.name].shape[:3]
        output_image_shape = spatial_shape + (n_channels,)
        empty_image = np.zeros(output_image_shape, dtype=dtype)

        for layer in self.reader.preprocessors:
            if isinstance(layer, PadLayer):
                empty_image, _ = layer(empty_image)

        if self.fill_constant != 0.0:
            empty_image[:] = self.fill_constant

        return empty_image

    def _save_current_image(self):
        if self.input_image is None:
            return

        output_name = self.reader.get_subject_id(self.image_id)
        self.writer(self.image_out, output_name, self.input_image[self.name])
