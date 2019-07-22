# -*- coding: utf-8 -*-
"""
windows aggregator decode sampling grid coordinates and image id from
batch data, forms image level output and write to hard drive.
"""
from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict

import numpy as np
import pandas as pd

# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-branches
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
                 postfix='niftynet_out',
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
        """
        Function used to save multiple outputs listed in the window
        dictionary. For the fields that have the keyword 'window' in the
        dictionary key, it will be saved as image. The rest will be saved as
        csv. CSV files will contain at saving a first line of 0 (to be
        changed into the header by the user), the first column being the
        index of the window, followed by the list of output and the location
        array for each considered window

        :param window: dictionary of output
        :param location: location of the input
        :return:
        """
        n_samples = location.shape[0]
        location_cropped = {}
        for key in window:
            if 'window' in key:  # all outputs to be created as images should
                # contained the keyword "window"
                window[key], location_cropped[key] = self.crop_batch(
                    window[key], location, self.window_border)

        for batch_id in range(n_samples):
            image_id = location[batch_id, 0]
            if image_id != self.image_id:
                # image name changed:
                # save current result and create an empty result file
                self._save_current_image()
                self._save_current_csv()
                if self._is_stopping_signal(location[batch_id]):
                    return False

                self.image_out, self.csv_out = {}, {}
                for key in window:
                    if 'window' in key:
                        # to be saved as image
                        self.image_out[key] = self._initialise_empty_image(
                            image_id=image_id,
                            n_channels=window[key].shape[-1],
                            dtype=window[key].dtype)
                    else:
                        # to be saved as csv file
                        n_elements = np.int64(
                            np.asarray(window[key]).size / n_samples)
                        table_header = [
                            '{}_{}'.format(key, idx)
                            for idx in range(n_elements)
                        ] if n_elements > 1 else ['{}'.format(key)]
                        table_header += [
                            'coord_{}'.format(idx)
                            for idx in range(location.shape[-1])
                        ]
                        self.csv_out[key] = self._initialise_empty_csv(
                            key_names=table_header)

            for key in window:
                if 'window' in key:
                    x_start, y_start, z_start, x_end, y_end, z_end = \
                        location_cropped[key][batch_id, 1:]
                    self.image_out[key][
                        x_start:x_end, y_start:y_end, z_start:z_end, ...] = \
                        window[key][batch_id, ...]
                else:
                    window[key] = np.asarray(window[key]).reshape(
                        [n_samples, -1])
                    window_save = window[key][batch_id:batch_id + 1, :]
                    window_loc = location[batch_id:batch_id + 1, :]
                    csv_row = np.concatenate([window_save, window_loc], 1)
                    csv_row = csv_row.ravel()
                    key_names = self.csv_out[key].columns
                    self.csv_out[key] = self.csv_out[key].append(
                        OrderedDict(zip(key_names, csv_row)),
                        ignore_index=True)
        return True

    def _initialise_empty_image(self, image_id, n_channels, dtype=np.float):
        """
        Initialise an empty image in which to populate the output
        :param image_id: image_id to be used in the reader
        :param n_channels: numbers of channels of the saved output (for
        multimodal output)
        :param dtype: datatype used for the saving
        :return: the initialised empty image
        """
        self.image_id = image_id
        spatial_shape = self.input_image[self.name].shape[:3]
        output_image_shape = spatial_shape + (n_channels, )
        empty_image = np.zeros(output_image_shape, dtype=dtype)
        for layer in self.reader.preprocessors:
            if isinstance(layer, PadLayer):
                empty_image, _ = layer(empty_image)

        if self.fill_constant != 0.0:
            empty_image[:] = self.fill_constant

        return empty_image

    def _initialise_empty_csv(self, key_names):
        """
        Initialise a csv output file with a first line of zeros

        :param n_channel: number of saved fields
        :return: empty first line of the array to be saved as csv
        """
        return pd.DataFrame(columns=key_names)

    def _save_current_image(self):
        """
        For all the outputs to be saved as images, go through the dictionary
        and save the resulting output after reversing the initial preprocessing
        :return:
        """
        if self.input_image is None:
            return
        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, PadLayer):
                for i in self.image_out:
                    self.image_out[i], _ = layer.inverse_op(self.image_out[i])
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                for i in self.image_out:
                    self.image_out[i], _ = layer.inverse_op(self.image_out[i])
        subject_name = self.reader.get_subject_id(self.image_id)
        for i in self.image_out:
            filename = "{}_{}_{}.nii.gz".format(i, subject_name, self.postfix)
            source_image_obj = self.input_image[self.name]
            misc_io.save_data_array(self.output_path, filename,
                                    self.image_out[i], source_image_obj,
                                    self.output_interp_order)
            self.log_inferred(subject_name, filename)
        return

    def _save_current_csv(self):
        """
        For all output to be saved as csv, loop through the dictionary of
        output and create the csv
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
