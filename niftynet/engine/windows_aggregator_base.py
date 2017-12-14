# -*- coding: utf-8 -*-
"""
This module is used to cache window-based network outputs,
form a image-level output,
write the cached the results to hard drive.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf


class ImageWindowsAggregator(object):
    """
    Image windows are retrieved and analysed by the
    tensorflow graph, this windows aggregator receives
    output window data in numpy array. To access image-level
    information the reader is needed.
    """

    def __init__(self, image_reader=None):
        self.reader = image_reader
        self._image_id = None

    @property
    def input_image(self):
        """
        Get the corresponding input image of these batch data.
        So that the batch data can be stored correctly in
        terms of interpolation order, orientation, pixdims.

        :return: an image object from image reader
        """
        if self.image_id is not None and self.reader:
            return self.reader.output_list[self.image_id]
        return None

    @property
    def image_id(self):
        """
        Index of the image in the output image list maintained by
        image reader.

        :return: integer of the position in image list
        """
        return self._image_id

    @image_id.setter
    def image_id(self, current_id):
        try:
            self._image_id = int(current_id)
        except (ValueError, TypeError):
            tf.logging.fatal("unknown image id format (should be an integer)")

    def decode_batch(self, *args, **kwargs):
        """
        The implementation of caching and writing batch output
        goes here. This function should return False when the
        location vector is stopping signal, to notify the
        inference loop to terminate.

        :param args:
        :param kwargs:
        :return: True if more batch data are expected, False otherwise
        """
        raise NotImplementedError

    @staticmethod
    def _is_stopping_signal(location_vector):
        return np.any(location_vector < 0)

    @staticmethod
    def crop_batch(window, location, border):
        """
        This utility function removes two borders along each
        spatial dim of the output image window data,
        adjusts window spatial coordinates accordingly.

        :param window:
        :param location:
        :param border:
        :return:
        """
        if border == ():
            border = (0, 0, 0)
        assert len(border) == 3, \
            "unknown border format (should be an array of" \
            "three elements corresponding to 3 spatial dims"

        window_shape = window.shape
        spatial_shape = window_shape[1:-1]
        n_spatial = len(spatial_shape)
        for idx in range(n_spatial):
            location[:, idx + 1] = location[:, idx + 1] + border[idx]
            location[:, idx + 4] = location[:, idx + 4] - border[idx]
        if np.any(location < 0):
            return window, location

        cropped_shape = np.max(location[:, 4:7] - location[:, 1:4], axis=0)
        left = np.floor(
            (spatial_shape - cropped_shape[:n_spatial])/2.0).astype(np.int)
        if np.any(left < 0):
            tf.logging.fatal(
                'network output window can be '
                'cropped by specifying the border parameter in config file, '
                'but here the output window %s is already smaller '
                'than the input window size minus padding: %s, '
                'not supported by this aggregator',
                spatial_shape, cropped_shape)
            raise ValueError
        if n_spatial == 1:
            window = window[:,
                            left[0]:(left[0] + cropped_shape[0]),
                            np.newaxis, np.newaxis, ...]
        elif n_spatial == 2:
            window = window[:,
                            left[0]:(left[0] + cropped_shape[0]),
                            left[1]:(left[1] + cropped_shape[1]),
                            np.newaxis, ...]
        elif n_spatial == 3:
            window = window[:,
                            left[0]:(left[0] + cropped_shape[0]),
                            left[1]:(left[1] + cropped_shape[1]),
                            left[2]:(left[2] + cropped_shape[2]),
                            ...]
        else:
            tf.logging.fatal(
                'unknown output format: shape %s'
                ' spatial dims are: %s', window_shape, spatial_shape)
            raise NotImplementedError
        return window, location
