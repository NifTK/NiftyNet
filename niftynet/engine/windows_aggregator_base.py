# -*- coding: utf-8 -*-
"""
This module is used to cache window-based network outputs,
 form a image-level output,
 write the cached the results to hard drive
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
        terms of interpolation order, orientation, pixdims
        :return: an image object from image reader
        """
        if self.image_id is not None and self.reader:
            return self.reader.output_list[self.image_id]
        return None

    @property
    def image_id(self):
        return self._image_id

    @image_id.setter
    def image_id(self, current_id):
        try:
            self._image_id = int(current_id)
        except ValueError:
            tf.logging.fatal("unknown image id format (should be an integer")

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
            return window, location
        assert len(border) == 3, \
            "unknown border format (should be an array of" \
            "three elements corresponding to 3 spatial dims"

        window_shape = window.shape
        if len(window_shape) != 5:
            raise NotImplementedError(
                "window shape not supported: {}".format(window_shape))
        spatial_shape = window_shape[1:4]
        assert all([win_size > 2 * border_size
                    for (win_size, border_size)
                    in zip(spatial_shape, border)]), \
            "window sizes should be larger than inference border size * 2"
        window = window[:,
                 border[0]:spatial_shape[0] - border[0],
                 border[1]:spatial_shape[1] - border[1],
                 border[2]:spatial_shape[2] - border[2], ...]
        for idx in range(3):
            location[:, idx + 1] = location[:, idx + 1] + border[idx]
            location[:, idx + 4] = location[:, idx + 4] - border[idx]
        return window, location
