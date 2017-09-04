# -*- coding: utf-8 -*-
"""This module loads images from csv files and outputs numpy arrays"""
from __future__ import absolute_import, division, print_function

from copy import deepcopy

import numpy as np
import tensorflow as tf
from six import string_types

import niftynet.utilities.util_csv as util_csv
from niftynet.io.image_type import ImageFactory
from niftynet.layer.base_layer import Layer, DataDependentLayer, RandomisedLayer
from niftynet.utilities.user_parameters_helper import make_input_tuple
from niftynet.utilities.util_common import print_progress_bar

# NP_TF_DTYPES = {'i': tf.int32, 'u': tf.int32, 'b': tf.int32, 'f': tf.float32}
NP_TF_DTYPES = {'i': tf.float32,
                'u': tf.float32,
                'b': tf.float32,
                'f': tf.float32}


def infer_tf_dtypes(image_array):
    return NP_TF_DTYPES.get(image_array.dtype[0].kind, tf.float32)


class ImageReader(Layer):
    """
    For a concrete example:
    _input_sources define multiple modality mappings, e.g.,
    _input_sources {'image': ('T1', 'T2'),
                    'label': ('manual_map',)}
    means
    'image' consists of two components, formed by
    concatenating 'T1' and 'T2' input source images.
    'label' consists of one component, loading from 'manual_map'

    self._names: a tuple of the output names of this reader.
    ('image', 'labels')

    self._shapes: the shapes after combining input sources
    {'image': (192, 160, 192, 1, 2), 'label': (192, 160, 192, 1, 1)}

    self._dtypes: store the dictionary of tensorflow shapes
    {'image': tf.float32, 'label': tf.float32}

    self.output_list is a list of dictionaries, with each item:
    {'image': <niftynet.io.image_type.SpatialImage4D object>,
     'label': <niftynet.io.image_type.SpatialImage3D object>}
    """

    def __init__(self, names):
        # list of file names
        self._file_list = None
        self._input_sources = None
        self._shapes = None
        self._dtypes = None
        self._names = None
        self.names = names

        # list of image objects
        self.output_list = None
        self.current_id = -1

        self.preprocessors = []
        super(ImageReader, self).__init__(name='image_reader')

    def initialise_reader(self, data_param, task_param):
        """
        task_param specifies how to combine user input modalities
        e.g., for multimodal segmentation 'image' corresponds to multiple
        modality sections, 'label' corresponds to one modality section
        """
        if not self.names:
            tf.logging.fatal('Please specify data names, this should '
                             'be a subset of SUPPORTED_INPUT provided '
                             'in application file')
            raise ValueError
        self._names = [name for name in self.names
                       if vars(task_param).get(name, None)]

        self._input_sources = {name: vars(task_param).get(name)
                               for name in self.names}
        data_to_load = {}
        for name in self._names:
            for source in self._input_sources[name]:
                try:
                    data_to_load[source] = data_param[source]
                except KeyError:
                    tf.logging.fatal(
                        'reader name [%s] requires [%s], however it is not '
                        'specified as a section in the config, '
                        'current input section names: %s',
                        name, source, list(data_param))
                    raise ValueError

        self._file_list = util_csv.load_and_merge_csv_files(data_to_load)
        self.output_list = _filename_to_image_list(
            self._file_list, self._input_sources, data_param)
        for name in self.names:
            tf.logging.info(
                'image reader: loading [%s] from %s (%d)',
                name, self.input_sources[name], len(self.output_list))

    def prepare_preprocessors(self):
        for layer in self.preprocessors:
            if isinstance(layer, DataDependentLayer):
                layer.train(self.output_list)

    def add_preprocessing_layers(self, layers):
        assert self.output_list is not None, \
            'Please initialise the reader first, ' \
            'before adding preprocessors.'
        if isinstance(layers, Layer):
            self.preprocessors.append(layers)
        else:
            self.preprocessors.extend(layers)
        self.prepare_preprocessors()

    # pylint: disable=arguments-differ
    def layer_op(self, idx=None, shuffle=True):
        """
        this layer returns a dictionary
          keys: self.output_fields
          values: image volume array
        """
        if idx is None and shuffle:
            # training, with random list output
            idx = np.random.randint(len(self.output_list))

        if idx is None and not shuffle:
            # testing, with sequential output
            # accessing self.current_id, not suitable for multi-thread
            idx = self.current_id + 1
            self.current_id = idx

        try:
            idx = int(idx)
        except ValueError:
            idx = -1

        if idx < 0 or idx >= len(self.output_list):
            return -1, None, None

        image_dict = self.output_list[idx]
        image_data_dict = {field: image.get_data()
                           for (field, image) in image_dict.items()}
        interp_order_dict = {field: image.interp_order
                             for (field, image) in image_dict.items()}
        if self.preprocessors:
            preprocessors = [deepcopy(layer) for layer in self.preprocessors]
            # dictionary of masks is cached
            mask = None
            for layer in preprocessors:
                # import time; local_time = time.time()
                if layer is None:
                    continue
                if isinstance(layer, RandomisedLayer):
                    layer.randomise()
                    image_data_dict = layer(image_data_dict, interp_order_dict)
                else:
                    image_data_dict, mask = layer(image_data_dict, mask)
                # print('%s, %.3f sec'%(layer, -local_time + time.time()))
        return idx, image_data_dict, interp_order_dict

    @property
    def shapes(self):
        """
        image shapes before any preprocessing
        :return: tuple of integers as image shape
        """
        # to have fast access, the spatial dimensions are not accurate
        # 1) only read from the first image in list
        # 2) not considering effects of random augmentation layers
        # but time and modality dimensions should be correct
        if not self.output_list:
            tf.logging.fatal("please initialise the reader first")
            raise RuntimeError
        if not self._shapes:
            first_image = self.output_list[0]
            self._shapes = {field: first_image[field].shape
                            for field in self.names}
        return self._shapes

    @property
    def tf_dtypes(self):
        if not self.output_list:
            tf.logging.fatal("please initialise the reader first")
            raise RuntimeError
        if not self._dtypes:
            first_image = self.output_list[0]
            self._dtypes = {field: infer_tf_dtypes(first_image[field])
                            for field in self.names}
        return self._dtypes

    @property
    def input_sources(self):
        if not self._input_sources:
            tf.logging.fatal("please initialise the reader first")
            raise RuntimeError
        return self._input_sources

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, fields_tuple):
        # output_fields is a sequence of output names
        # each name might correspond to a list of multiple input sources
        # this should be specified in CUSTOM section in the config
        self._names = make_input_tuple(fields_tuple, string_types)

    def get_subject_id(self, image_index):
        return self._file_list.iloc[image_index, 0]


def _filename_to_image_list(file_list, mod_dict, data_param):
    """
    converting a list of filenames to a list of image objects
    useful properties (e.g. interp_order) are added to each object
    """
    volume_list = []
    for idx in range(len(file_list)):
        print_progress_bar(idx, len(file_list),
                           prefix='reading datasets headers',
                           decimals=1, length=10, fill='*')
        # combine fieldnames and volumes as a dictionary
        _dict = {field: _create_image(file_list, idx, modalities, data_param)
                 for (field, modalities) in mod_dict.items()}
        volume_list.append(_dict)
    return volume_list


def _create_image(file_list, idx, modalities, data_param):
    """
    data_param consists of description of each modality
    This function combines modalities according to the 'modalities'
    parameter and create <niftynet.io.input_type.SpatialImage*D>
    """
    try:
        file_path = tuple(file_list.loc[idx, mod] for mod in modalities)
        interp_order = tuple(data_param[mod].interp_order for mod in modalities)
        pixdim = tuple(data_param[mod].pixdim for mod in modalities)
        axcodes = tuple(data_param[mod].axcodes for mod in modalities)
    except KeyError:
        tf.logging.fatal(
            "Specified modality names %s "
            "not found in config: input sections %s",
            modalities, list(data_param))
        raise
    except AttributeError:
        tf.logging.fatal(
            'data params must contain: interp_order, pixdim, axcodes')
        raise

    image_properties = {'file_path': file_path,
                        'name': modalities,
                        'interp_order': interp_order,
                        'output_pixdim': pixdim,
                        'output_axcodes': axcodes}
    return ImageFactory.create_instance(**image_properties)
