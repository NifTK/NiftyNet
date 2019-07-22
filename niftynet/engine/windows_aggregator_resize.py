# -*- coding: utf-8 -*-
"""
Windows aggregator resize each item
in a batch output and save as an image.
"""
from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict

import numpy as np
import pandas as pd

import niftynet.io.misc_io as misc_io
from niftynet.engine.sampler_resize_v2 import zoom_3d
from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.pad import PadLayer


class ResizeSamplesAggregator(ImageWindowsAggregator):
    """
    This class decodes each item in a batch by resizing each image
    window and save as a new image volume. Multiple output image can be
    proposed and csv output can be performed as well
    """

    def __init__(self,
                 image_reader,
                 name='image',
                 output_path=os.path.join('.', 'output'),
                 window_border=(),
                 interp_order=0,
                 postfix='niftynet_out'):
        ImageWindowsAggregator.__init__(
            self, image_reader=image_reader, output_path=output_path)
        self.name = name
        self.image_out = None
        self.csv_out = None
        self.window_border = window_border
        self.output_interp_order = interp_order
        self.postfix = postfix
        self.current_out = {}

    def decode_batch(self, window, location):
        """
        Resizing each output image window in the batch as an image volume
        location specifies the original input image (so that the
        interpolation order, original shape information retained in the

        generated outputs).For the fields that have the keyword 'window' in the
        dictionary key, it will be saved as image. The rest will be saved as
        csv. CSV files will contain at saving a first line of 0 (to be
        changed into the header by the user), the first column being the
        index of the window, followed by the list of output.

        """
        n_samples = location.shape[0]
        for batch_id in range(n_samples):
            if self._is_stopping_signal(location[batch_id]):
                return False
            self.image_id = location[batch_id, 0]
            self.image_out, self.csv_out = {}, {}
            for key in window:
                if 'window' in key:
                    # saving image output
                    while window[key].ndim < 5:
                        window[key] = window[key][..., np.newaxis, :]
                    self.image_out[key] = window[key][batch_id, ...]
                else:
                    # saving csv output
                    window[key] = np.asarray(window[key]).reshape(
                        [n_samples, -1])
                    n_elements = window[key].shape[-1]
                    table_header = [
                        '{}_{}'.format(key, idx) for idx in range(n_elements)
                    ] if n_elements > 1 else ['{}'.format(key)]
                    self.csv_out[key] = self._initialise_empty_csv(
                        key_names=table_header)
                    csv_row = window[key][batch_id:batch_id + 1, :].ravel()
                    self.csv_out[key] = self.csv_out[key].append(
                        OrderedDict(zip(table_header, csv_row)),
                        ignore_index=True)
            self._save_current_image()
            self._save_current_csv()

        return True

    def _initialise_image_shape(self, image_id, n_channels):
        """
        Return the shape of the empty image to be saved
        :param image_id: index to find the appropriate input image from the
        reader
        :param n_channels: number of channels of the image
        :return:  shape of the empty image
        """
        self.image_id = image_id
        spatial_shape = self.input_image[self.name].shape[:3]
        output_image_shape = spatial_shape + (1, n_channels)
        empty_image = np.zeros(output_image_shape, dtype=np.bool)
        for layer in self.reader.preprocessors:
            if isinstance(layer, PadLayer):
                empty_image, _ = layer(empty_image)
        return empty_image.shape

    def _save_current_image(self):
        """
        Loop through the dictionary of images output and resize and reverse
        the preprocessing prior to saving
        :return:
        """
        if self.input_image is None:
            return

        self.current_out = {}
        for i in self.image_out:
            resize_to_shape = self._initialise_image_shape(
                image_id=self.image_id, n_channels=self.image_out[i].shape[-1])
            window_shape = resize_to_shape
            current_out = self.image_out[i]
            while current_out.ndim < 5:
                current_out = current_out[..., np.newaxis, :]
            if self.window_border and any([b > 0 for b in self.window_border]):
                np_border = self.window_border
                while len(np_border) < 5:
                    np_border = np_border + (0, )
                np_border = [(b, ) for b in np_border]
                current_out = np.pad(current_out, np_border, mode='edge')
            image_shape = current_out.shape
            zoom_ratio = \
                [float(p) / float(d) for p, d in zip(window_shape, image_shape)]
            image_shape = list(image_shape[:3]) + [1, image_shape[-1]]
            current_out = np.reshape(current_out, image_shape)
            current_out = zoom_3d(
                image=current_out,
                ratio=zoom_ratio,
                interp_order=self.output_interp_order)
            self.current_out[i] = current_out

        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, PadLayer):
                for i in self.image_out:
                    self.current_out[i], _ = layer.inverse_op(
                        self.current_out[i])
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                for i in self.image_out:
                    self.image_out[i], _ = layer.inverse_op(self.image_out[i])
        subject_name = self.reader.get_subject_id(self.image_id)
        for i in self.image_out:
            filename = "{}_{}_{}.nii.gz".format(i, subject_name, self.postfix)
            source_image_obj = self.input_image[self.name]
            misc_io.save_data_array(self.output_path, filename,
                                    self.current_out[i], source_image_obj,
                                    self.output_interp_order)
            self.log_inferred(subject_name, filename)
        return

    def _save_current_csv(self):
        """
        Save all csv output present in the dictionary of csv_output.
        :return:
        """
        if self.input_image is None:
            return
        subject_name = self.reader.get_subject_id(self.image_id)
        for i in self.csv_out:
            filename = "{}_{}_{}.csv".format(i, subject_name, self.postfix)
            misc_io.save_csv_array(self.output_path, filename, self.csv_out[i])
            self.log_inferred(subject_name, filename)
        return

    def _initialise_empty_csv(self, key_names):
        """
        Initialise the array to be saved as csv as a line of zeros according
        to the number of elements to be saved
        :param n_channel:
        :return:
        """
        return pd.DataFrame(columns=key_names)
