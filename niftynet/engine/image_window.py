# -*- coding: utf-8 -*-
"""
This module provides an interface for data elements passed
from sampler to network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

N_SPATIAL = 3
LOCATION_FORMAT = "{}_location"
BUFFER_POSITION_DTYPE = tf.int32


# TF_NP_DTYPES = {tf.int32: np.int32, tf.float32: np.float32}


class ImageWindow(object):
    """
    Each window is associated with a tuple of coordinates.
    These data properties are used to create TF
    placeholders when constructing a TF graph. Samplers
    read the data specifications and fill the placeholder
    with data.
    """

    def __init__(self, names, shapes, dtypes):
        self.names = names
        self.shapes = shapes
        self.dtypes = dtypes
        self.n_samples = 1
        self.has_dynamic_shapes = self._check_dynamic_shapes()
        self._placeholders_dict = None

    @classmethod
    def from_data_reader_properties(cls,
                                    source_names,
                                    image_shapes,
                                    image_dtypes,
                                    data_param):
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
            data_param={
                'modality1': ParserNamespace(spatial_window_size=(10, 10, 2)),
                'modality2': ParserNamespace(spatial_window_size=(10, 10, 2)),
                'modality3': ParserNamespace(spatial_window_size=(5, 5, 1))}

        see ``niftynet.io.ImageReader`` for more details.

        :param source_names: input image names
        :param image_shapes: tuple of image window shapes
        :param image_dtypes: tuple of image window data types
        :param data_param: dict of each input source specifications
        :return: an ImageWindow instance
        """
        try:
            input_names = tuple(source_names)
        except TypeError:
            tf.logging.fatal('image names should be a dictionary of strings')
            raise
        try:
            # complete window shapes based on user input and input_image sizes
            spatial_shapes = {
                name: _read_window_sizes(modalities, data_param)
                for (name, modalities) in source_names.items()}
            shapes = {
                name: _complete_partial_window_sizes(
                    spatial_shapes[name], image_shapes[name])
                for name in input_names}
        except KeyError:
            tf.logging.fatal('data_param wrong format %s', data_param)
            raise
        # create ImageWindow instance
        return cls(names=input_names,
                   shapes=shapes,
                   dtypes=image_dtypes)

    def set_spatial_shape(self, spatial_window):
        """
        Overrides all spatial window defined in input modalities sections
        this is useful when do inference with a spatial window
        which is different from the training specifications.

        :param spatial_window: tuple of integers specifying new shape
        :return:
        """
        try:
            spatial_window = [int(win_size) for win_size in spatial_window]
        except ValueError:
            tf.logging.fatal("spatial window should be an array of int")
            raise
        self.shapes = {
            name: _complete_partial_window_sizes(
                spatial_window, self.shapes[name])
            for name in self.names}
        # update based on the latest spatial shapes
        self.has_dynamic_shapes = self._check_dynamic_shapes()
        if self._placeholders_dict is not None:
            self._update_placeholders_dict(n_samples=self.n_samples)

    def _update_placeholders_dict(self, n_samples=1):
        # batch size=1 if the shapes are dynamic
        self.n_samples = 1 if self.has_dynamic_shapes else n_samples

        names = list(self.names)
        placeholders = []
        try:
            placeholders = [
                tf.placeholder(
                    dtype=self.dtypes[name],
                    shape=[self.n_samples] + list(self.shapes[name]),
                    name=name)
                for name in names]
        except TypeError:
            tf.logging.fatal(
                'shape should be defined as dict of iterable %s', self.shapes)
            raise
        # extending names with names of coordinates
        names.extend([LOCATION_FORMAT.format(name) for name in names])
        # extending placeholders with names of coordinates
        location_shape = [self.n_samples, 1 + N_SPATIAL * 2]
        placeholders.extend(
            [tf.placeholder(dtype=BUFFER_POSITION_DTYPE,
                            shape=location_shape,
                            name=name)
             for name in self.names])
        self._placeholders_dict = dict(zip(names, placeholders))

    def placeholders_dict(self, n_samples=1):
        """
        This function create a dictionary with items of
        ``{name: placeholders}``
        name should match the queue input names
        placeholders corresponds to the image window data
        for each of these items an additional ``{location_name: placeholders}``
        is created to hold the spatial location of the image window

        :param n_samples: specifies the number of image windows
        :return: a dictionary with window data and locations placeholders
        """

        if self._placeholders_dict is not None:
            return self._placeholders_dict
        self._update_placeholders_dict(n_samples)
        return self._placeholders_dict

    def coordinates_placeholder(self, name):
        """
        get coordinates placeholder, location name is formed
        using ``LOCATION_FORMAT``

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
        get the image data placeholder by name

        :param name: input name string
        :return: image placeholder
        """
        try:
            return self._placeholders_dict[name]
        except TypeError:
            tf.logging.fatal('call placeholders_dict to initialise first')
            raise

    def _check_dynamic_shapes(self):
        """
        Check whether the shape of the window is fully specified

        :return: True indicates it's dynamic, False indicates
         the window size is fully specified.
        """
        for shape in list(self.shapes.values()):
            try:
                for dim_length in shape:
                    if not dim_length:
                        return True
            except TypeError:
                return False
        return False

    def match_image_shapes(self, image_shapes):
        """
        if the window has dynamic shapes, this function
        infers the fully specified shape from the image_shapes.

        :param image_shapes:
        :return: dict of fully specified window shapes
        """
        if self.has_dynamic_shapes:
            static_window_shapes = self.shapes.copy()
            # fill the None element in dynamic shapes using image_sizes
            for name in self.names:
                static_window_shapes[name] = tuple(
                    win_size if win_size else image_shape
                    for (win_size, image_shape) in
                    zip(list(self.shapes[name]), image_shapes[name]))
        else:
            static_window_shapes = self.shapes
        return static_window_shapes


def _read_window_sizes(input_mod_list, input_data_param):
    """
    Read window_size from config dict
    group them based on output names,
    this function ensures that in the multimodality case
    the spatial window sizes are the same across modalities.

    :param input_mod_list: list of input source names
    :param input_data_param: input source properties obtained
        by parameters parser
    :return: spatial window size
    """
    try:
        window_sizes = [input_data_param[input_name].spatial_window_size
                        for input_name in input_mod_list]
    except (AttributeError, TypeError, KeyError):
        tf.logging.fatal('unknown input_data_param format %s %s',
                         input_mod_list, input_data_param)
        raise
    if not all(window_sizes):
        window_sizes = [win_size for win_size in window_sizes if win_size]
    uniq_window_set = set(window_sizes)
    if len(uniq_window_set) > 1:
        # pylint: disable=logging-format-interpolation
        tf.logging.fatal(
            "trying to combine input sources "
            "with different window sizes: %s", window_sizes)
        raise NotImplementedError
    window_shape = None
    if uniq_window_set:
        window_shape = uniq_window_set.pop()
    try:
        return tuple(int(win_size) for win_size in window_shape)
    except (TypeError, ValueError):
        pass
    try:
        # try to make it a tuple
        return int(window_shape),
    except (TypeError, ValueError):
        tf.logging.fatal('unknown spatial_window_size param %s, %s',
                         input_mod_list, input_data_param)
        raise


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
