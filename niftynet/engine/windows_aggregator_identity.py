# -*- coding: utf-8 -*-
"""
windows aggregator saves each item in a batch output as an image.
"""
from __future__ import absolute_import, print_function, division

import os
import numpy as np

import niftynet.io.misc_io as misc_io
from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator


class WindowAsImageAggregator(ImageWindowsAggregator):
    """
    This class saves each item in a batch output to images,
    the output filenames can be defined in three ways:

        1. location is None (input image from a random distribution):
        a uuid is generated as output filename.

        2. the length of the location array is 2:
        (indicates output image is from an
        interpolation of two input images):

                - ``location[batch_id, 0]`` is used as a ``base_name``,
                - ``location[batch_id, 0]`` is used as a ``relative_id``

        output file name is ``"{}_{}"%(base_name, relative_id)``.

        3. the length of the location array is greater than 2:
        (indicates output image is from single input image)
        ``location[batch_id, 0]`` is used as the file name
    """
    def __init__(self,
                 image_reader=None,
                 output_path=os.path.join('.', 'output'),
                 postfix='_niftynet_generated'):
        ImageWindowsAggregator.__init__(
            self, image_reader=image_reader, output_path=output_path)
        self.output_path = os.path.abspath(output_path)
        self.inferred_csv = os.path.join(self.output_path, 'inferred.csv')
        self.output_id = {'base_name': None, 'relative_id': 0}
        self.postfix = postfix
        if os.path.exists(self.inferred_csv):
            os.remove(self.inferred_csv)

    def _decode_subject_name(self, location=None):
        if self.reader:
            image_id = int(location)
            return self.reader.get_subject_id(image_id)
        import uuid
        return str(uuid.uuid4())

    def decode_batch(self, window, location, name_opt=''):
        n_samples = location.shape[0]
        location_init = np.copy(location)
        test = None
        for w in window:
            if 'window' in w:
                # print(w)
                test = np.ones_like(window[w])
                # print(w, location_init, window[w].shape, self.window_border,
                #       np.sum(window[w]))
                window[w], _ = self.crop_batch(window[w], location_init, self.window_border)
                location_init = np.copy(location)
                print(w, np.sum(window[w]), np.max(window[w]))
        _, location = self.crop_batch(test, location_init, self.window_border)
        for batch_id in range(n_samples):
            image_id, x_start, y_start, z_start, x_end, y_end, z_end = \
                location[batch_id, :]
            if image_id != self.image_id:
                # image name changed:
                #    save current image and create an empty image
                self._save_current_image(name_opt)
                self._save_current_csv(name_opt)
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
                        self.csv_out[w] = self._initialise_empty_csv(
                            image_id=image_id,n_channel=window[w][0].shape[
                                                             -1]+location_init[0,:].shape[-1])
            for w in window:
                if 'window' in w:
                    self.image_out[w][x_start:x_end,
                           y_start:y_end,
                           z_start:z_end, ...] = window[w][batch_id, ...]
                else:
                    window_loc = np.concatenate([window[w],
                                                np.tile(location_init[
                                                            batch_id,...],
                                                        [window[w].shape[0],1])],1)
                    self.csv_out[w] = np.concatenate([self.csv_out[w],
                                                      window_loc],0)
        return True

    def decode_batch_old(self, window, location=None):
        if location is not None:
            n_samples = location.shape[0]
            for batch_id in range(n_samples):
                if self._is_stopping_signal(location[batch_id]):
                    return False
                filename = self._decode_subject_name(location[batch_id, 0])
                # if base file name changed, reset relative name index
                if filename != self.output_id['base_name']:
                    self.output_id['base_name'] = filename
                    self.output_id['relative_id'] = 0
                # when location has two component, the name should
                # be constructed as a composite of two input filenames
                if len(location[batch_id]) == 2:
                    output_name = '{}_{}'.format(
                        self.output_id['base_name'],
                        self._decode_subject_name(location[batch_id, 1]))
                else:
                    output_name = self.output_id['base_name']
                self._save_current_image(self.output_id['relative_id'],
                                         output_name,
                                         window[batch_id, ...])
                self.output_id['relative_id'] += 1
            return True
        n_samples = window.shape[0]
        for batch_id in range(n_samples):
            filename = self._decode_subject_name()
            self._save_current_image(
                batch_id, filename, window[batch_id, ...])
        return False

    def _save_current_image(self, idx, filename, image):
        if image is None:
            return
        uniq_name = "{}_{}{}.nii.gz".format(idx, filename, self.postfix)
        misc_io.save_data_array(self.output_path, uniq_name, image, None)
        with open(self.inferred_csv, 'a') as csv_file:
            filename = os.path.join(self.output_path, filename)
            csv_file.write('{},{}\n'.format(idx, uniq_name))
        return

    def _save_current_csv(self, name_opt):
        if self.input_image is None:
            return
        subject_name = self.reader.get_subject_id(self.image_id)
        for i in self.csv_out:
            filename = "{}_{}_niftynet_out.csv".format(i+name_opt,subject_name)
            misc_io.save_csv_array(self.output_path,
                                filename,
                                self.csv_out[i])
            self.log_inferred(subject_name, filename)
        return