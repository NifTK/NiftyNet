# -*- coding: utf-8 -*-
"""
windows aggregator resize each item
in a batch output and save as an image
"""
from __future__ import absolute_import, print_function, division

import os

import numpy as np

import niftynet.io.misc_io as misc_io
from niftynet.engine.sampler_resize import zoom_3d
from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer


class ResizeSamplesAggregator(ImageWindowsAggregator):
    def __init__(self,
                 image_reader,
                 name='image',
                 output_path=os.path.join('.', 'output'),
                 window_border=(),
                 interp_order=0):
        ImageWindowsAggregator.__init__(self, image_reader=image_reader)
        self.name = name
        self.output_path = os.path.abspath(output_path)
        self.window_border = window_border
        self.output_interp_order = interp_order

    def decode_batch(self, window, location):
        n_samples = location.shape[0]
        window, location = self.crop_batch(window, location, self.window_border)
        for batch_id in range(n_samples):
            if self._is_stopping_signal(location[batch_id]):
                return False
            self.image_id, _, _, _, _, _, _ = location[batch_id, :]
            self.image_ref = self._initialise_empty_image(
                image_id=self.image_id,
                n_channels=window.shape[-1],
                dtype=window.dtype)
            self._save_current_image(window[batch_id, ...])

        return True

    def _initialise_empty_image(self, image_id, n_channels, dtype=np.float):
        self.image_id = image_id
        spatial_shape = self.input_image[self.name].shape[:3]
        output_image_shape = spatial_shape + (1, n_channels,)
        empty_image = np.zeros(output_image_shape, dtype=dtype)

        return empty_image

    def _save_current_image(self, image_out):
        if self.input_image is None:
            return

        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                image_out, _ = layer.inverse_op(image_out)
        window_shape = self.input_image[self.name].shape
        while image_out.ndim < 5:
            image_out = image_out[..., np.newaxis, :]
        image_shape = image_out.shape
        zoom_ratio = [p / d for p, d in zip(window_shape, image_shape)]
        image_shape = list(image_shape[:3]) + [1, image_shape[-1]]
        image_out = np.reshape(image_out, image_shape)

        self.image_ref[...] = zoom_3d(
            image=image_out,
            ratio=zoom_ratio,
            interp_order=self.output_interp_order)
        subject_name = self.reader.get_subject_id(self.image_id)
        filename = "{}_niftynet_out.nii.gz".format(subject_name)
        source_image_obj = self.input_image[self.name]
        misc_io.save_data_array(self.output_path,
                                filename,
                                self.image_ref,
                                source_image_obj,
                                self.output_interp_order)
        return
