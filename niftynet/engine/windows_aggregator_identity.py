# -*- coding: utf-8 -*-
"""
windows aggregator saves each item in a batch output as an image
"""
from __future__ import absolute_import, print_function, division

import os

import niftynet.io.misc_io as misc_io
from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator


class WindowAsImageAggregator(ImageWindowsAggregator):
    def __init__(self,
                 image_reader=None,
                 output_path=os.path.join('.', 'output')):
        ImageWindowsAggregator.__init__(self, image_reader=image_reader)
        self.output_path = os.path.abspath(output_path)
        self.output_id = {'base_name': None, 'relative_id': 0}

    def decode_subject_name(self, location=None):
        if self.reader:
            image_id = location
            return self.reader.get_subject_id(image_id)
        else:
            import uuid
            return uuid.uuid4()

    def decode_batch(self, window, location=None):
        if location is not None:
            n_samples = location.shape[0]
            for batch_id in range(n_samples):
                if self._is_stopping_signal(location[batch_id]):
                    return False
                filename = self.decode_subject_name(location[batch_id, 0])
                # if base file name changed, reset relative name index
                if filename != self.output_id['base_name']:
                    self.output_id['base_name'] = filename
                    self.output_id['relative_id'] = 0
                # when location has two component, the name should
                # be constructed as a composite of two input filenames
                if len(location[batch_id]) == 2:
                    output_name = '{}_{}'.format(
                        self.output_id['base_name'],
                        self.decode_subject_name(location[batch_id, 1]))
                else:
                    output_name = self.output_id['base_name']
                self._save_current_image(self.output_id['relative_id'],
                                         output_name,
                                         window[batch_id, ...])
                self.output_id['relative_id'] += 1
            return True
        else:
            n_samples = window.shape[0]
            for batch_id in range(n_samples):
                filename = self.decode_subject_name()
                self._save_current_image(
                    batch_id, filename, window[batch_id, ...])
            return False

    def _save_current_image(self, idx, filename, image):
        if image is None:
            return
        uniq_name = "{}_{}_niftynet_generated.nii.gz".format(idx, filename)
        misc_io.save_data_array(self.output_path, uniq_name, image, None)
        return
