# -*- coding: utf-8 -*-
"""This module loads images from csv files and outputs numpy arrays."""
from __future__ import absolute_import, division, print_function

from copy import deepcopy

import numpy as np
import pandas
import tensorflow as tf
from six import string_types

from niftynet.io.image_sets_partitioner import COLUMN_UNIQ_ID
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
    """
    Choosing a suitable tf dtype based on the dtype of input numpy array.
    """
    return NP_TF_DTYPES.get(image_array.dtype[0].kind, tf.float32)


class ImageReader(Layer):
    """
    For a concrete example::

        _input_sources define multiple modality mappings, e.g.,
        _input_sources {'image': ('T1', 'T2'), 'label': ('manual_map',)}

    means:

    'image' consists of two components, formed by
    concatenating 'T1' and 'T2' input source images.
    'label' consists of one component, loading from 'manual_map'

    :param self._names: a tuple of the output names of this reader.
        ``('image', 'labels')``

    :param self._shapes: the shapes after combining input sources
        ``{'image': (192, 160, 192, 1, 2), 'label': (192, 160, 192, 1, 1)}``

    :param self._dtypes: store the dictionary of tensorflow shapes
        ``{'image': tf.float32, 'label': tf.float32}``

    :param self.output_list: a list of dictionaries, with each item::

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

    def initialise(self, data_param, task_param, file_list):
        """
        ``task_param`` specifies how to combine user input modalities.
        e.g., for multimodal segmentation 'image' corresponds to multiple
        modality sections, 'label' corresponds to one modality section

        This function converts elements of ``file_list`` into
        dictionaries of image objects, and save them to ``self.output_list``.
        """
        if not self.names:
            tf.logging.fatal('Please specify data input keywords, this should '
                             'be a subset of SUPPORTED_INPUT provided '
                             'in application file.')
            raise ValueError
        filtered_names = [name for name in self.names
                          if vars(task_param).get(name, None)]
        if not filtered_names:
            tf.logging.fatal("Reader requires task input keywords %s, but "
                             "not exist in the config file.\n"
                             "Available task keywords: %s",
                             filtered_names, list(vars(task_param)))
            raise ValueError

        self._names = filtered_names
        self._input_sources = dict((name, vars(task_param).get(name))
                                   for name in self.names)
        required_sections = \
            sum([list(vars(task_param).get(name)) for name in self.names], [])

        for required in required_sections:
            try:
                if (file_list is None) or \
                        (required not in list(file_list)) or \
                        (file_list[required].isnull().all()):
                    tf.logging.fatal('Reader required input section '
                                     'name [%s], but in the filename list '
                                     'the column is empty.', required)
                    raise ValueError
            except (AttributeError, TypeError, ValueError):
                tf.logging.fatal(
                    'file_list parameter should be a '
                    'pandas.DataFrame instance and has input '
                    'section name [%s] as a column name.', required)
                if required_sections:
                    tf.logging.fatal('Reader requires section(s): %s',
                                     required_sections)
                if file_list is not None:
                    tf.logging.fatal('Configuration input sections are: %s',
                                     list(file_list))
                raise

        self._file_list = file_list
        self.output_list = _filename_to_image_list(
            self._file_list, self._input_sources, data_param)
        for name in self.names:
            tf.logging.info(
                'Image reader: loading [%s] from %s (%d)',
                name, self.input_sources[name], len(self.output_list))

    def prepare_preprocessors(self):
        """
        Some preprocessors requires an initial step to initialise
        data dependent internal parameters.

        This function find these preprocessors and run the initialisations.
        """
        for layer in self.preprocessors:
            if isinstance(layer, DataDependentLayer):
                layer.train(self.output_list)

    def add_preprocessing_layers(self, layers):
        """
        Adding a ``niftynet.layer`` or a list of layers as preprocessing steps.
        """
        assert self.output_list is not None, \
            'Please initialise the reader first, ' \
            'before adding preprocessors.'
        if isinstance(layers, Layer):
            self.preprocessors.append(layers)
        else:
            self.preprocessors.extend(layers)
        self.prepare_preprocessors()

    # pylint: disable=arguments-differ,too-many-branches
    def layer_op(self, idx=None, shuffle=True):
        """
        this layer returns dictionaries::

            keys: self.output_fields
            values: image volume array

        """
        if idx is None:
            if shuffle:
                # training, with random list output
                idx = np.random.randint(len(self.output_list))
            else:
                # testing, with sequential output
                # accessing self.current_id, not suitable for multi-thread
                idx = self.current_id + 1
                self.current_id = idx

        try:
            image_dict = self.output_list[idx]
        except (IndexError, TypeError):
            return -1, None, None

        image_data_dict = \
            {field: image.get_data() for (field, image) in image_dict.items()}
        interp_order_dict = \
            {field: image.interp_order for (field, image) in image_dict.items()}

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
        Image shapes before any preprocessing.

        :return: tuple of integers as image shape


        .. caution::

            To have fast access, the spatial dimensions are not accurate

                1. only read from the first image in list
                2. not considering effects of random augmentation layers
                    but time and modality dimensions should be correct
        """
        if not self.output_list:
            tf.logging.fatal("Please initialise the reader first.")
            raise RuntimeError
        if not self._shapes:
            first_image = self.output_list[0]
            self._shapes = {field: first_image[field].shape
                            for field in self.names}
        return self._shapes

    @property
    def tf_dtypes(self):
        """
        Infer input data dtypes in TF
        (using the first image in the file list).
        """
        if not self.output_list:
            tf.logging.fatal("Please initialise the reader first.")
            raise RuntimeError
        if not self._dtypes:
            first_image = self.output_list[0]
            self._dtypes = {field: infer_tf_dtypes(first_image[field])
                            for field in self.names}
        return self._dtypes

    @property
    def input_sources(self):
        """
        returns mapping of input keywords and input sections
        e.g., input_sources::

            {'image': ('T1', 'T2'),
             'label': ('manual_map',)}

        map task parameter keywords ``image`` and ``label`` to
        section names ``T1``, ``T2``, and ``manual_map`` respectively.
        """
        if not self._input_sources:
            tf.logging.fatal("Please initialise the reader first.")
            raise RuntimeError
        return self._input_sources

    @property
    def names(self):
        """

        :return: the keys of ``self.input_sources`` dictionary
        """
        return self._names

    @names.setter
    def names(self, fields_tuple):
        """
        output_fields is a sequence of output names
        each name might correspond to a list of multiple input sources
        this should be specified in CUSTOM section in the config
        """
        self._names = make_input_tuple(fields_tuple, string_types)

    def get_subject_id(self, image_index):
        """
        Given an integer id returns the subject id.
        """
        try:
            return self._file_list.iloc[image_index][COLUMN_UNIQ_ID]
        except KeyError:
            tf.logging.warning('Unknown subject id in reader table.')
            raise


def _filename_to_image_list(file_list, mod_dict, data_param):
    """
    Converting a list of filenames to a list of image objects,
    Properties (e.g. interp_order) are added to each object
    """
    volume_list = []
    for idx in range(len(file_list)):
        # create image instance for each subject
        print_progress_bar(idx, len(file_list),
                           prefix='reading datasets headers',
                           decimals=1, length=10, fill='*')

        # combine fieldnames and volumes as a dictionary
        _dict = {}
        for field, modalities in mod_dict.items():
            _dict[field] = _create_image(file_list, idx, modalities, data_param)

        # skipping the subject if there're missing image components
        if _dict and None not in list(_dict.values()):
            volume_list.append(_dict)

    if not volume_list:
        tf.logging.fatal(
            "Empty filename lists, please check the csv "
            "files. (removing csv_file keyword if it is in the config file "
            "to automatically search folders and generate new csv "
            "files again)\n\n"
            "Please note in the matched file names, each subject id are "
            "created by removing all keywords listed `filename_contains` "
            "in the config.\n\n"
            "E.g., `filename_contains=foo, bar` will match file "
            "foo_subject42_bar.nii.gz, and the subject id is _subject42_.")
        raise IOError
    return volume_list


def _create_image(file_list, idx, modalities, data_param):
    """
    data_param consists of description of each modality
    This function combines modalities according to the 'modalities'
    parameter and create <niftynet.io.input_type.SpatialImage*D>
    """
    try:
        file_path = tuple(file_list.loc[idx, mod] for mod in modalities)
        any_missing = any([pandas.isnull(file_name) or not bool(file_name)
                           for file_name in file_path])
        if any_missing:
            # todo: enable missing modalities again
            # the file_path of a multimodal image will contain `nan`, e.g.
            # this should be handled by `ImageFactory.create_instance`
            # ('testT1.nii.gz', 'testT2.nii.gz', nan, 'testFlair.nii.gz')
            return None
        interp_order = tuple(data_param[mod].interp_order
                             for mod in modalities)
        pixdim = tuple(data_param[mod].pixdim for mod in modalities)
        axcodes = tuple(data_param[mod].axcodes for mod in modalities)
    except KeyError:
        tf.logging.fatal(
            "Specified modality names %s "
            "not found in config: input sections %s.",
            modalities, list(data_param))
        raise
    except AttributeError:
        tf.logging.fatal(
            "Data params must contain: interp_order, pixdim, axcodes.\n"
            "Reader must be initialised with a dataframe as file_list.")
        raise

    image_properties = {'file_path': file_path,
                        'name': modalities,
                        'interp_order': interp_order,
                        'output_pixdim': pixdim,
                        'output_axcodes': axcodes}
    return ImageFactory.create_instance(**image_properties)
