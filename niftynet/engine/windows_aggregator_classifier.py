# -*- coding: utf-8 -*-
"""
windows aggregator resize each item
in a batch output and save as an image
"""
from __future__ import absolute_import, print_function, division

import os

import numpy as np

import niftynet.io.misc_io as misc_io
from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer


class ClassifierSamplesAggregator(ImageWindowsAggregator):
    """
    This class decodes each item in a batch by saving classification
    labels to a new image volume.
    """
    def __init__(self,
                 image_reader,
                 name='image',
                 output_path=os.path.join('.', 'output'),
                 postfix='_niftynet_out'):
        ImageWindowsAggregator.__init__(
            self, image_reader=image_reader, output_path=output_path)
        self.name = name
        self.output_interp_order = 0
        self.postfix = postfix
        self.csv_path = os.path.join(self.output_path, self.postfix+'.csv')
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def decode_batch_old(self, window, location):
        """
        window holds the classifier labels
        location is a holdover from segmentation and may be removed
        in a later refactoring, but currently hold info about the stopping
        signal from the sampler
        """
        n_samples = window.shape[0]
        for batch_id in range(n_samples):
            if self._is_stopping_signal(location[batch_id]):
                return False
            self.image_id = location[batch_id, 0]
            self._save_current_image(window[batch_id, ...])
        return True

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



    def _save_current_image(self, image_out):
        if self.input_image is None:
            return
        window_shape = [1, 1, 1, 1, image_out.shape[-1]]
        image_out = np.reshape(image_out, window_shape)
        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                image_out, _ = layer.inverse_op(image_out)
        subject_name = self.reader.get_subject_id(self.image_id)
        filename = "{}{}.nii.gz".format(subject_name, self.postfix)
        source_image_obj = self.input_image[self.name]
        misc_io.save_data_array(self.output_path,
                                filename,
                                image_out,
                                source_image_obj,
                                self.output_interp_order)
        with open(self.csv_path, 'a') as csv_file:
            data_str = ','.join([str(i) for i in image_out[0, 0, 0, 0, :]])
            csv_file.write(subject_name+','+data_str+'\n')
        self.log_inferred(subject_name, filename)
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

