# -*- coding: utf-8 -*-
"""This module loads images from csv files and outputs numpy arrays."""
from __future__ import absolute_import, division, print_function

import argparse
from abc import ABCMeta, abstractmethod, abstractproperty
from copy import deepcopy

import numpy as np
import tensorflow as tf

from niftynet.io.misc_io import dtype_casting
from niftynet.layer.base_layer import (DataDependentLayer, Layer,
                                       RandomisedLayer)
from niftynet.utilities.util_common import ParserNamespace, look_up_operations

DEFAULT_INTERP_ORDER = 1
SUPPORTED_DATA_SPEC = {
    'csv_file', 'path_to_search', 'filename_contains', 'filename_not_contains',
    'filename_removefromid', 'interp_order', 'loader', 'pixdim', 'axcodes',
    'spatial_window_size'
}


def infer_tf_dtypes(image_array):
    """
    Choosing a suitable tf dtype based on the dtype of input numpy array.
    """
    return dtype_casting(
        image_array.dtype[0], image_array.interp_order[0], as_tf=True)


class BaseImageSource(Layer):
    """
    Base class for F/S image reader and other image-input sources.
    """

    __metaclass__ = ABCMeta

    def __init__(self, name='image_source'):
        super(BaseImageSource, self).__init__(name=name)

        self._spatial_ranks = None
        self._shapes = None
        self._dtypes = None

        self.current_id = -1
        self.preprocessors = []

    @staticmethod
    def _get_section_input_sources(task_param, section_names):
        """
        Filters a list of section names against a task_param
        struct eliminating any invalid section names. Throws an
        exception if no valid entries are found in the list.

        :param task_param: task_param dictionary of application
        specific settings.
        :param section_names: list of image specification sections.
        :return: the filtered list of section names and the dictionary
        of corresponding modalities.
        """

        if not isinstance(task_param, dict):
            task_param = vars(task_param)

        valid_names = [
            name for name in section_names if task_param.get(name, None)
        ]
        if not valid_names:
            tf.logging.fatal(
                "Reader requires task input keywords %s, but "
                "not exist in the config file.\n"
                "Available task keywords: %s", section_names, list(task_param))
            raise ValueError

        modalities = {name: task_param.get(name) for name in valid_names}

        return valid_names, modalities

    @abstractmethod
    def _load_spatial_ranks(self):
        """
        :return: loads the spatial rank dict, returned by spatial_ranks
        """
        raise NotImplementedError

    @property
    def spatial_ranks(self):
        """
        :return: the shapes of the images in the collections provided
        by this source as dict of integers with image source names as keys.
        """

        if not self._spatial_ranks:
            self._spatial_ranks = self._load_spatial_ranks()

        return self._spatial_ranks

    @abstractmethod
    def _load_shapes(self):
        """
        :return: the dict of image shapes returned by shapes
        """
        raise NotImplementedError

    @property
    def shapes(self):
        """
        Image shapes before any preprocessing.

        :return: for every image source, the tuple of integers as image shape


        .. caution::

            To have fast access, the spatial dimensions are not accurate

                1. only read from the first image in list
                2. not considering effects of random augmentation layers
                    but time and modality dimensions should be correct
        """

        if not self._shapes:
            self._shapes = self._load_shapes()

        return self._shapes

    def prepare_preprocessors(self):
        """
        Some preprocessors requires an initial step to initialise
        data dependent internal parameters.

        This function find these preprocessors and run the initialisations.
        """

        for layer in self.preprocessors:
            if isinstance(layer, DataDependentLayer):
                layer.train((self.get_output_image(i)
                             for i in range(self.num_subjects)),
                            num_subjects=self.num_subjects)

    def add_preprocessing_layers(self, layers):
        """
        Adding a ``niftynet.layer`` or a list of layers as preprocessing steps.
        """
        assert self.num_subjects > 0, \
            'Please initialise the reader first, ' \
            'before adding preprocessors.'
        if isinstance(layers, Layer):
            self.preprocessors.append(layers)
        else:
            self.preprocessors.extend(layers)
        self.prepare_preprocessors()

    @abstractmethod
    def _load_dtypes(self):
        """
        :return: the dict of tensorflow data types returned by tf_dtypes
        """
        raise NotImplementedError

    @property
    def tf_dtypes(self):
        """
        Infer input data dtypes in TF
        (using the first image in the file list).
        """

        if not self._dtypes:
            self._dtypes = self._load_dtypes()

        return self._dtypes

    @abstractproperty
    def names(self):
        """
        :return: the list of input source names
        """
        raise NotImplementedError

    @abstractproperty
    def num_subjects(self):
        """
        :return the total number of subjects across the collections.
        """
        raise NotImplementedError

    @abstractmethod
    def get_subject_id(self, image_index):
        """
        Given an integer id returns the subject id.
        """
        raise NotImplementedError

    @abstractproperty
    def input_sources(self):
        """
        returns mapping of input keywords and input sections
        e.g., input_sources::

            {'image': ('T1', 'T2'),
             'label': ('manual_map',)}

        map task parameter keywords ``image`` and ``label`` to
        section names ``T1``, ``T2``, and ``manual_map`` respectively.
        """
        raise NotImplementedError

    @abstractmethod
    def get_image_index(self, subject_id):
        """
        Given a subject id, return the file_list index
        :param subject_id: a string with the subject id
        :return: an int with the file list index
        """
        raise NotImplementedError

    @abstractmethod
    def _get_image_and_interp_dict(self, idx):
        """
        Given an index this function must produce two dictionaries
        containing one image data tensor and one interpolation
        order for every named image collection

        On error: E.g., when the index is out of bounds, this function
        should return a None for both dictionaries.

        :return: one dictionary containing image data and one dictionary
        containing interpolation orders.
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_image(self, idx):
        """
        :return: the i-th image (including meta data) outputted by
        this source as a input-source/image dictionary.
        """
        raise NotImplementedError

    # pylint: disable=arguments-differ
    def layer_op(self, idx=None, shuffle=True):
        """
        this layer returns dictionaries::

            keys: self.output_fields
            values: image volume array

        """
        if idx is None:
            if shuffle:
                # training, with random list output
                idx = np.random.randint(self.num_subjects)
            else:
                # testing, with sequential output
                # accessing self.current_id, not suitable for multi-thread
                idx = self.current_id + 1
                self.current_id = idx

        image_data_dict, interp_order_dict = \
            self._get_image_and_interp_dict(idx)

        if not image_data_dict:
            idx = -1
        else:
            preprocessors = [deepcopy(layer) for layer in self.preprocessors]
            # dictionary of masks is cached
            mask = None
            for layer in preprocessors:
                # import time; local_time = time.time()
                if layer is None:
                    continue
                if isinstance(layer, RandomisedLayer):
                    if "random_elastic_deformation" not in layer.name:
                        layer.randomise()
                    else:
                        layer.randomise(image_data_dict)

                    image_data_dict = layer(image_data_dict, interp_order_dict)
                elif isinstance(layer, Layer):
                    image_data_dict, mask = layer(image_data_dict, mask)
                    # print('%s, %.3f sec'%(layer, -local_time + time.time()))

        return idx, image_data_dict, interp_order_dict


def param_to_dict(input_data_param):
    """
    Validate the user input ``input_data_param``
    raise an error if it's invalid.

    :param input_data_param:
    :return: input data specifications as a nested dictionary
    """
    error_msg = 'Unknown ``data_param`` type. ' \
                'It should be a nested dictionary: '\
                '{"modality_name": {"input_property": value}} '\
                'or a dictionary of: {"modality_name": '\
                'niftynet.utilities.util_common.ParserNamespace}'
    data_param = deepcopy(input_data_param)
    if isinstance(data_param, (ParserNamespace, argparse.Namespace)):
        data_param = vars(data_param)
    if not isinstance(data_param, dict):
        raise ValueError(error_msg)
    for mod in data_param:
        mod_param = data_param[mod]
        if isinstance(mod_param, (ParserNamespace, argparse.Namespace)):
            dict_param = vars(mod_param)
        elif isinstance(mod_param, dict):
            dict_param = mod_param
        else:
            raise ValueError(error_msg)
        for data_key in dict_param:
            look_up_operations(data_key, SUPPORTED_DATA_SPEC)
        data_param[mod] = dict_param
    return data_param
