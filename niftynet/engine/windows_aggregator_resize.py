# -*- coding: utf-8 -*-
"""
Windows aggregator resize each item
in a batch output and save as an image.
"""
from __future__ import absolute_import, print_function, division

import os

import numpy as np

import niftynet.io.misc_io as misc_io
from niftynet.engine.sampler_resize_v2 import zoom_3d
from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.pad import PadLayer


class ResizeSamplesAggregator(ImageWindowsAggregator):
    """
    This class decodes each item in a batch by resizing each image
    window and save as a new image volume.
    """
    def __init__(self,
                 image_reader,
                 name='image',
                 output_path=os.path.join('.', 'output'),
                 window_border=(),
                 interp_order=0,
                 postfix='_niftynet_out'):
        ImageWindowsAggregator.__init__(
            self, image_reader=image_reader, output_path=output_path)
        self.name = name
        self.window_border = window_border
        self.output_interp_order = interp_order
        self.postfix = postfix

    def decode_batch(self, window, location):
        """
        Resizing each output image window in the batch as an image volume
        location specifies the original input image (so that the
        interpolation order, original shape information retained in the
        generated outputs).
        """
        n_samples = location.shape[0]
        window, location = self.crop_batch(window, location, self.window_border)
        for batch_id in range(n_samples):
            if self._is_stopping_signal(location[batch_id]):
                return False
            self.image_id = location[batch_id, 0]
            resize_to_shape = self._initialise_image_shape(
                image_id=self.image_id,
                n_channels=window.shape[-1])
            self._save_current_image(window[batch_id, ...], resize_to_shape)
        return True

    def _initialise_image_shape(self, image_id, n_channels):
        self.image_id = image_id
        spatial_shape = self.input_image[self.name].shape[:3]
        output_image_shape = spatial_shape + (1, n_channels,)
        empty_image = np.zeros(output_image_shape, dtype=np.bool)
        for layer in self.reader.preprocessors:
            if isinstance(layer, PadLayer):
                empty_image, _ = layer(empty_image)
        return empty_image.shape

    def _save_current_image(self, image_out, resize_to):
        if self.input_image is None:
            return
        window_shape = resize_to
        while image_out.ndim < 5:
            image_out = image_out[..., np.newaxis, :]
        if self.window_border and any([b > 0 for b in self.window_border]):
            np_border = self.window_border
            while len(np_border) < 5:
                np_border = np_border + (0,)
            np_border = [(b,) for b in np_border]
            image_out = np.pad(image_out, np_border, mode='edge')
        image_shape = image_out.shape
        zoom_ratio = \
            [float(p) / float(d) for p, d in zip(window_shape, image_shape)]
        image_shape = list(image_shape[:3]) + [1, image_shape[-1]]
        image_out = np.reshape(image_out, image_shape)
        image_out = zoom_3d(image=image_out,
                            ratio=zoom_ratio,
                            interp_order=self.output_interp_order)

        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, PadLayer):
                image_out, _ = layer.inverse_op(image_out)
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
        self.log_inferred(subject_name, filename)
        return
