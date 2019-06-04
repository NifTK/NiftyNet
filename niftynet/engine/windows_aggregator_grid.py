# -*- coding: utf-8 -*-
"""
windows aggregator decode sampling grid coordinates and image id from
batch data, forms image level output and write to hard drive.
"""
from __future__ import absolute_import, print_function, division

import os

import numpy as np

import niftynet.io.misc_io as misc_io
from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.pad import PadLayer


class GridSamplesAggregator(ImageWindowsAggregator):
    """
    This class keeps record of the currently cached image,
    initialised as all zeros, and the values are replaced
    by image window data decoded from batch.
    """
    def __init__(self,
                 image_reader,
                 name='image',
                 output_path=os.path.join('.', 'output'),
                 window_border=(),
                 interp_order=0,
                 postfix='_niftynet_out',
                 fill_constant=0.0):
        ImageWindowsAggregator.__init__(
            self, image_reader=image_reader, output_path=output_path)
        self.name = name
        self.image_out = None
        self.csv_out = None
        self.window_border = window_border
        self.output_interp_order = interp_order
        self.postfix = postfix
        self.fill_constant = fill_constant

    def decode_batch(self, window, location):
        n_samples = location.shape[0]
        location_init = np.copy(location)
        init_ones = None
        for w in window:
            if 'window' in w: # all outputs to be created as images should
                # contained the keyword "window"
                init_ones = np.ones_like(window[w])
                window[w], _ = self.crop_batch(window[w], location_init, self.window_border)
                location_init = np.copy(location)
                print(w, np.sum(window[w]), np.max(window[w]))
        _, location = self.crop_batch(init_ones, location_init,
                                      self.window_border)
        for batch_id in range(n_samples):
            image_id, x_start, y_start, z_start, x_end, y_end, z_end = \
                location[batch_id, :]
            if image_id != self.image_id:
                # image name changed:
                #    save current image and create an empty image
                self._save_current_image()
                self._save_current_csv()
                if self._is_stopping_signal(location[batch_id]):
                    return False
                self.image_out = {}
                self.csv_out = {}
                for w in window:
                    if 'window' in w:
                        self.image_out[w] = self._initialise_empty_image(
                            image_id=image_id,
                            n_channels=window[w].shape[-1],
                            dtype=window[w].dtype)
                    else:
                        if isinstance(window[w],(np.int,np.float32,np.bool)):
                            self.csv_out[w] = self._initialise_empty_csv(1+ location_init[0,
                                                        :].shape[-1])
                        else:
                            self.csv_out[w] = self._initialise_empty_csv(
                                            n_channel=window[w][0].shape[-1]
                                                      + location_init[0,
                                                        :].shape[-1])
            for w in window:
                if 'window' in w:
                    self.image_out[w][x_start:x_end,
                        y_start:y_end,
                        z_start:z_end, ...] = window[w][batch_id, ...]
                else:
                    if not isinstance(window[w], (int, np.float32, bool)):
                        window_loc = np.concatenate([window[w],
                                                np.tile(location_init[
                                                            batch_id, ...],
                                                        [window[w].shape[0],1])],1)
                    else:
                        window_loc = np.concatenate([np.reshape(window[w],[1,
                                                                           1]),
                                                     np.tile(location_init[
                                                                 batch_id, ...],
                                                             [1, 1])], 1)
                    self.csv_out[w] = np.concatenate([self.csv_out[w],
                                                      window_loc],0)
        return True

    def decode_batch_old(self, window, location):
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

    def _initialise_empty_csv(self, n_channel):
        return np.zeros([1, n_channel])

    def _save_current_image(self):
        if self.input_image is None:
            return
        for i in self.image_out:
            print(np.sum(self.image_out[i]), " is sum of image out %s before"
                  % i)
        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, PadLayer):
                for i in self.image_out:
                    self.image_out[i], _ = layer.inverse_op(self.image_out[i])
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                for i in self.image_out:
                    self.image_out[i], _ = layer.inverse_op(self.image_out[i])
        subject_name = self.reader.get_subject_id(self.image_id)
        for i in self.image_out:
            print(np.sum(self.image_out[i]), " is sum of image out %s after"
                  % i)
        for i in self.image_out:
            filename = "{}_{}_{}.nii.gz".format(i, subject_name, self.postfix)
            source_image_obj = self.input_image[self.name]
            misc_io.save_data_array(self.output_path,
                                    filename,
                                    self.image_out[i],
                                    source_image_obj,
                                    self.output_interp_order)
            self.log_inferred(subject_name, filename)
        return

    def _save_current_image_old(self):
        if self.input_image is None:
            return

        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, PadLayer):
                self.image_out, _ = layer.inverse_op(self.image_out)
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                self.image_out, _ = layer.inverse_op(self.image_out)
        subject_name = self.reader.get_subject_id(self.image_id)
        filename = "{}_{}.nii.gz".format(subject_name, self.postfix)
        source_image_obj = self.input_image[self.name]
        misc_io.save_data_array(self.output_path,
                                filename,
                                self.image_out,
                                source_image_obj,
                                self.output_interp_order)
        self.log_inferred(subject_name, filename)
        return

    def _save_current_csv(self):
        if self.input_image is None:
            return
        subject_name = self.reader.get_subject_id(self.image_id)
        for i in self.csv_out:
            filename = "{}_{}_{}.csv".format(i, subject_name, self.postfix)
            misc_io.save_csv_array(self.output_path, filename, self.csv_out[i])
            self.log_inferred(subject_name, filename)
        return
