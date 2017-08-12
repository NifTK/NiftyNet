# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
from copy import deepcopy

import numpy as np
import pandas
import tensorflow as tf

from niftynet.io.input_type import ImageFactory
from niftynet.layer.base_layer import Layer, DataDependentLayer, RandomisedLayer
from niftynet.utilities.misc_common import printProgressBar
from niftynet.utilities.user_parameters_helper import make_input_tuple

NP_TF_DTYPES = {'i': tf.int32, 'u': tf.int32, 'b': tf.int32, 'f': tf.float32}


def infer_tf_dtypes(image_array):
    return NP_TF_DTYPES.get(image_array.dtype.kind, tf.float32)


class ImageReader(Layer):
    def __init__(self, output_fields):
        # list of file names
        self._file_list = None
        self._input_sources = None
        self._shapes = None
        self._dtypes = None
        self._output_fields = output_fields

        # cache the first image data array for shape/data type info
        self.__first_image = None


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
        app_type = task_param.name
        self._file_list = ImageReader.load_and_merge_csv_files(data_param)

        if app_type == "net_segment.py":
            from niftynet.application.segmentation_application \
                import SUPPORTED_INPUT
            # only choose fields that are supported by the application
            # (SUPPORTED_INPUT) and have user parameter specification
        elif app_type == "net_gan.py":
            from niftynet.application.gan_application \
                import SUPPORTED_INPUT

        if not self.output_fields:
            # by default, reader tries to output all supported fields
            self.output_fields = SUPPORTED_INPUT
        self._output_fields = [field_name for field_name in self.output_fields
                               if vars(task_param).get(field_name)]
        self._input_sources = {field: vars(task_param).get(field, None)
                               for field in self.output_fields}
        self.output_list = filename_to_image_list(self._file_list,
                                                  self._input_sources,
                                                  data_param)
        for field in self.output_fields:
            tf.logging.info('image reader: loading [{}] from {} ({})'.format(
                field, self.input_sources[field], len(self.output_list)))

    def prepare_preprocessors(self):
        for layer in self.preprocessors:
            if isinstance(layer, DataDependentLayer):
                layer.train(self.output_list)

    def add_preprocessing_layers(self, layers):
        if isinstance(layers, Layer):
            self.preprocessors.append(layers)
        else:
            self.preprocessors.extend(layers)
        self.prepare_preprocessors()

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
            #  accessing self.current_id, not suitable for multi-thread
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
            # dictionary of mask is cached
            mask = None
            for layer in preprocessors:
                if layer is None:
                    continue
                if isinstance(layer, RandomisedLayer):
                    layer.randomise()
                    image_data_dict = layer(image_data_dict, interp_order_dict)
                else:
                    image_data_dict, mask = layer(image_data_dict, mask)
        return idx, image_data_dict, interp_order_dict

    @staticmethod
    def load_and_merge_csv_files(data_param):
        """
        Converts a list of csv_files in data_param
        in to a joint list of file names (by matching the first column)
        This function returns a <pandas.core.frame.DataFrame> of the
        joint list
        """
        _file_list = None
        for modality_name in data_param:
            csv_file = data_param.get(modality_name, '').csv_file
            if not os.path.isfile(csv_file):
                tf.logging.fatal("csv file {} not found.".format(csv_file))
                raise IOError
            csv_list = pandas.read_csv(
                csv_file, header=None, names=['subject_id', modality_name])
            if _file_list is None:
                _file_list = csv_list
                continue

            # merge _file_list based on subject_ids (first column of each csv)
            n_rows = _file_list.shape[0]
            _file_list = pandas.merge(_file_list, csv_list, on='subject_id')
            if _file_list.shape[0] != n_rows:
                tf.logging.warning("rows not matched in {}".format(csv_file))
        if _file_list.size == 0:
            tf.logging.fatal("no common subject_ids in filename lists,"
                             "please check the csv files.")
            raise IOError
        return _file_list

    @property
    def shapes(self):
        if not self.output_list:
            tf.logging.fatal("please initialise the reader first")
            raise RuntimeError
        if not self._shapes:
            if self.__first_image is None:
                _, self.__first_image, _ = self(idx=0)
            self._shapes = {field: self.__first_image[field].shape
                            for field in self.output_fields}
        return self._shapes

    @property
    def tf_dtypes(self):
        if not self.output_list:
            tf.logging.fatal("please initialise the reader first")
            raise RuntimeError
        if not self._dtypes:
            if self.__first_image is None:
                _, self.__first_image, _ = self(idx=0)
            self._dtypes = {field: infer_tf_dtypes(self.__first_image[field])
                            for field in self.output_fields}
        return self._dtypes

    @property
    def input_sources(self):
        if not self._input_sources:
            tf.logging.fatal("please initialise the reader first")
            raise RuntimeError
        return self._input_sources

    @property
    def output_fields(self):
        return self._output_fields

    @output_fields.setter
    def output_fields(self, fields_tuple):
        # output_fields is a sequence of output names
        # each name might correspond to a list of multiple input sources
        # this should be specified in CUSTOM section in the config
        self._output_fields = make_input_tuple(fields_tuple, basestring)


def filename_to_image_list(file_list, mod_dict, data_param):
    """
    converting a list of filenames to a list of image objects
    useful properties (e.g. interp_order) are added to each object
    """
    volume_list = []
    for idx in range(len(file_list)):
        printProgressBar(idx, len(file_list),
                         prefix='reading datasets headers',
                         decimals=1, length=10, fill='*')
        # combine fieldnames and volumes as a dictionary
        _dict = {field: create_image(file_list, idx, modalities, data_param)
                 for (field, modalities) in mod_dict.items()}
        volume_list.append(_dict)
    return volume_list


def create_image(file_list, idx, modalities, data_param):
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
            "Specified modality names {} "
            "not found in config: input sections {}".format(
                modalities, list(data_param)))
        raise
    image_properties = {'file_path': file_path,
                        'name': modalities,
                        'interp_order': interp_order,
                        'output_pixdim': pixdim,
                        'output_axcodes': axcodes}
    return ImageFactory.create_instance(**image_properties)
