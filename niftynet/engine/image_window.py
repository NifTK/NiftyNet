# -*- coding: utf-8 -*-
"""
This module provides an interface for data elements to be generated
by an image sampler.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy

import numpy as np
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.data.util import nest

from niftynet.utilities.util_common import ParserNamespace

N_SPATIAL = 3
LOCATION_FORMAT = "{}_location"
BUFFER_POSITION_NP_TYPE = np.int32
BUFFER_POSITION_DTYPE = tf.int32


class ImageWindow(object):
    """
    Each window is associated with a tuple of coordinates.
    These data properties are used to create TF
    placeholders or ``tf.data.Dataset`` when constructing a TF graph.
    Sampler read the data specifications and fill the placeholder/dataset.
    """

    def __init__(self, shapes, dtypes):
        """

        :param shapes: A nested structure of tuple
            corresponding to size of each image window
        :param dtypes: A nested structure of `tf.DType` objects
            corresponding to each image window
        """
        self._shapes = shapes
        self._dtypes = dtypes
        self._placeholders_dict = None

        self.n_samples = 1
        self.has_dynamic_shapes = self._check_dynamic_shapes()

    @property
    def names(self):
        """

        :return: a tuple of output modality names
        """
        return tuple(self._shapes)

    @property
    def shapes(self):
        """

        :return: a dictionary of image window and location shapes
        """
        shapes = {}
        for name in list(self._shapes):
            shapes[name] = tuple(
                [self.n_samples] + list(self._shapes[name]))
            shapes[LOCATION_FORMAT.format(name)] = tuple(
                [self.n_samples] + [1 + N_SPATIAL * 2])
        return shapes

    @property
    def tf_shapes(self):
        """
        :return: a dictionary of sampler output tensor shapes
        """
        output_shapes = nest.map_structure_up_to(
            self.tf_dtypes, tf.TensorShape, self.shapes)
        return output_shapes

    @property
    def tf_dtypes(self):
        """
        :return: tensorflow dtypes of the window.
        """
        dtypes = {}
        for name in list(self._dtypes):
            dtypes[name] = self._dtypes[name]
            dtypes[LOCATION_FORMAT.format(name)] = BUFFER_POSITION_DTYPE
        return dtypes

    @classmethod
    def from_data_reader_properties(cls,
                                    source_names,
                                    image_shapes,
                                    image_dtypes,
                                    window_sizes=None,
                                    allow_dynamic=False):
        """
        Create a window instance with input data properties
        each property is grouped into dict, with pairs of
        image_name: data_value. Some input images is a
        concatenated data array from multiple data sources.
        example of input::

            source_names={
                'image': (u'modality1', u'modality2'),
                'label': (u'modality3',)},
            image_shapes={
                'image': (192, 160, 192, 1, 2),
                'label': (192, 160, 192, 1, 1)},
            image_dtypes={
                'image': tf.float32,
                'label': tf.float32},
            window_sizes={
                'image': (10, 10, 2),
                'label': (10, 10, 2)}

        the ``window_sizes`` can also be::

            window_sizes={
                'modality1': (10, 10, 2),
                'modality3': (10, 10, 2)}

        or using a nested dictionary with 'spatial_window_size' (deprecating)::

            window_sizes={
                'modality1': {'spatial_window_size': (10, 10, 2)},
                'modality2': {'spatial_window_size': (10, 10, 2)},
                'modality3': {'spatial_window_size': (5, 5, 1)}}

        see ``niftynet.io.ImageReader`` for more details.

        :param source_names: input image names
        :param image_shapes: tuple of image window shapes
        :param image_dtypes: tuple of image window data types
        :param window_sizes: window sizes for the image image
        :param allow_dynamic: if True, window_sizes negative or 0 indicates
            dynamic window sizes; . Otherwise the dynamic sizes will be fixed
            as the image shapes; this assumes the same image size across the
            dataset.
        :return: an ImageWindow instance
        """
        try:
            image_shapes = nest.map_structure_up_to(
                image_dtypes, tuple, image_shapes)
        except KeyError:
            tf.logging.fatal('window_sizes wrong format %s', window_sizes)
            raise
        # create ImageWindow instance
        window_instance = cls(shapes=image_shapes, dtypes=image_dtypes)

        if not window_sizes:
            # image window sizes not specified, defaulting to image sizes.
            return window_instance

        window_instance.set_spatial_shape(window_sizes, source_names)
        if not allow_dynamic:
            full_shape = window_instance.match_image_shapes(image_shapes)
            window_instance.set_spatial_shape(full_shape)
        return window_instance

    def set_spatial_shape(self, spatial_window, source_names=None):
        """
        Set all spatial window of the window.

        spatial_window should be a dictionary of window sizes tuples
        or single window size tuple.  In the latter case the size
        will be used by all output image windows.

        :param spatial_window: tuple of integers specifying new shape
        :param source_names: list/dictionary of input source names
        :return:
        """
        win_sizes = copy.deepcopy(spatial_window)
        if isinstance(spatial_window, dict):
            for name in list(spatial_window):
                window_size = spatial_window[name]
                if isinstance(window_size,
                              (ParserNamespace, argparse.Namespace)):
                    window_size = vars(window_size)
                if not isinstance(window_size, dict):
                    win_sizes[name] = tuple(window_size)
                elif 'spatial_window_size' in window_size:
                    win_sizes[name] = tuple(
                        window_size['spatial_window_size'])
                else:
                    raise ValueError(
                        'window_sizes should be a nested dictionary')
        elif isinstance(spatial_window, (list, tuple)):
            # list or tuple of single window sizes
            win_sizes = {name: spatial_window for name in list(self._dtypes)}

        # complete window shapes based on user input and input_image sizes
        if source_names:
            spatial_shapes = _read_window_sizes(source_names, win_sizes)
        else:
            try:
                spatial_shapes = {}
                for name in list(self._dtypes):
                    spatial_shapes[name] = \
                        tuple(int(win_size) for win_size in win_sizes[name])
            except ValueError:
                tf.logging.fatal("spatial window should be an array of int")
                raise

        spatial_shapes = nest.map_structure_up_to(
            self._dtypes, tuple, spatial_shapes)

        self._shapes = {
            name: _complete_partial_window_sizes(spatial_shapes[name],
                                                 self._shapes[name])
            for name in list(self._shapes)}

        # update based on the latest spatial shapes
        self.has_dynamic_shapes = self._check_dynamic_shapes()
        if self._placeholders_dict is not None:
            self._update_placeholders_dict(n_samples=self.n_samples)

    def placeholders_dict(self, n_samples=1):
        """
        This function create a dictionary with items of
        ``{name: placeholders}``
        name should match the queue input names
        placeholders corresponds to the image window data
        for each of these items an additional ``{location_name: placeholders}``
        is created to hold the spatial location of the image window.
        Used in the queue-based tensorflow APIs.

        :param n_samples: specifies the number of image windows
        :return: a dictionary with window data and locations placeholders
        """

        if self._placeholders_dict is not None:
            return self._placeholders_dict
        self._update_placeholders_dict(n_samples)
        return self._placeholders_dict

    def coordinates_placeholder(self, name):
        """
        Get coordinates placeholder, location name is formed
        using ``LOCATION_FORMAT``.
        Used in the queue-based tensorflow APIs.

        :param name: input name string
        :return: coordinates placeholder
        """
        try:
            return self._placeholders_dict[LOCATION_FORMAT.format(name)]
        except TypeError:
            tf.logging.fatal('call placeholders_dict to initialise first')
            raise

    def image_data_placeholder(self, name):
        """
        Get the image data placeholder by name.
        Used in the queue-based tensorflow APIs.


        :param name: input name string
        :return: image placeholder
        """
        try:
            return self._placeholders_dict[name]
        except TypeError:
            tf.logging.fatal('call placeholders_dict to initialise first')
            raise

    def match_image_shapes(self, image_shapes):
        """
        If the window has dynamic shapes, this function
        infers the fully specified shape from the image_shapes.

        :param image_shapes:
        :return: dict of fully specified window shapes
        """
        if not self.has_dynamic_shapes:
            return self._shapes

        static_window_shapes = self._shapes.copy()
        # fill the None element in dynamic shapes using image_sizes
        for name in list(self._shapes):
            static_window_shapes[name] = tuple(
                win_size if win_size else image_shape
                for (win_size, image_shape) in
                zip(list(self._shapes[name]), image_shapes[name]))
        return static_window_shapes

    def _update_placeholders_dict(self, n_samples=1):
        """
        Update the placeholders according to the new n_samples (batch_size).
        Used in the queue-based tensorflow APIs.

        :param n_samples:
        :return:
        """
        # batch size=1 if the shapes are dynamic
        self.n_samples = 1 if self.has_dynamic_shapes else n_samples

        try:
            self._placeholders_dict = {}
            for name in list(self.tf_dtypes):
                self._placeholders_dict[name] = tf.placeholder(
                    dtype=self.tf_dtypes[name],
                    shape=self.shapes[name],
                    name=name)
        except TypeError:
            tf.logging.fatal(
                'shape should be defined as dict of iterable %s', self.shapes)
            raise

    def _check_dynamic_shapes(self):
        """
        Check whether the shape of the window is fully specified.

        :return: True indicates it's dynamic, False indicates
            the window size is fully specified.
        """
        for shape in list(self._shapes.values()):
            try:
                for dim_length in shape:
                    if not dim_length or dim_length < 0:
                        return True
            except TypeError:
                return False
        return False


def _read_window_sizes(input_mod_list, input_window_sizes):
    """
    Read window_size for each of the input image names defined
    by input_mod_list.keys().

    This function ensures that in the multimodality case
    the spatial window sizes are the same across modalities.
    For example::

    # the input indicates `image` is a concatenation of `mr` and `ct`.
    input_mod_list = {'image': ('mr', 'ct')}
    input_window_sizes = {'mr': (42, 42, 42)}
    returns: {'image': (42, 42, 42)}

    input_mod_list = ('image',)
    input_window_sizes = {'image': (42, 42, 42)}
    returns: {'image': (42, 42, 42)}

    input_mod_list = ('image',)
    input_window_sizes = (42, 42, 42)
    returns: {'image': (42, 42, 42)}

    # the input indicates a `image` and a `label` output.
    input_mod_list = ('image','label')
    input_window_sizes = (42, 42, 42)
    returns: {'image': (42, 42, 42), 'label': (42, 42, 42)}

    :param input_mod_list: list/dictionary of input source names
    :param input_window_sizes: input source properties obtained
        by parameters parser
    :return: {'output_name': spatial window size} dictionary
    """
    window_sizes = {}
    if isinstance(input_window_sizes, (tuple, list)):
        try:
            win_sizes = [int(win_size) for win_size in input_window_sizes]
        except ValueError:
            tf.logging.fatal("spatial window should be an array of int")
            raise
        # single window size for all inputs
        for name in set(input_mod_list):
            window_sizes[name] = win_sizes
        return window_sizes

    if isinstance(input_window_sizes, (ParserNamespace, argparse.Namespace)):
        input_window_sizes = vars(input_window_sizes)

    if not isinstance(input_window_sizes, dict):
        raise ValueError('window sizes should be a list/tuple/dictionary')

    output_names = set(input_mod_list)
    for name in output_names:
        window_size = None
        if name in input_window_sizes:
            # resolve output window size as input_window_sizes spec.
            window_size = input_window_sizes[name]
        for mod in input_mod_list[name]:
            # resolve output window size as input mod window size dict item
            if mod in input_window_sizes:
                window_size = input_window_sizes[mod]
        if not window_size:
            # input size not resolved
            raise ValueError('Unknown output window size '
                             'for input image {}'.format(name))
        if name in window_sizes:
            assert window_size == window_sizes[name], \
                "trying to use different window sizes for " \
                "the concatenated input {}".format(name)
        window_sizes[name] = window_size
    return window_sizes


def _complete_partial_window_sizes(win_size, img_size):
    """
    Window size can be partially specified in the config.
    This function complete the window size by making it
    the same ndim as img_size, and set the not added dim
    to size None. None values in window will be realised
    when each image is loaded.

    :param win_size: a tuple of (partial) window size
    :param img_size: a tuple of image size
    :return: a window size with the same ndim as image size,
        with None components to be inferred at runtime
    """
    img_ndims = len(img_size)
    # crop win_size list if it's longer than img_size
    win_size = list(win_size[:img_ndims])
    while len(win_size) < N_SPATIAL:
        win_size.append(-1)
    # complete win_size list if it's shorter than img_size
    while len(win_size) < img_ndims:
        win_size.append(img_size[len(win_size)])
    # replace zero with full length in the n-th dim of image
    win_size = [win if win > 0 else None for win in win_size]
    return tuple(win_size)
