# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas
import tensorflow as tf

from niftynet.io.input_type import ImageFactory
from niftynet.layer.base_layer import Layer
from niftynet.utilities.user_parameters_helper import make_input_tuple


class VolumeReader(Layer):
    def __init__(self, output_fields):
        # list of file names
        self._file_list = None

        self.output_fields = output_fields
        self._input_sources = None
        self._dtypes = None

        # list of image objects
        self.output_list = None
        self.current_id = None
        super(VolumeReader, self).__init__(name='volume_reader')

    @property
    def dtypes(self):
        if not self.output_list:
            tf.logging.fatal("please initialise the reader first")
            raise RuntimeError
        random_image = self.layer_op(shuffle=True)
        return {field: random_image[field].dtype
                for field in self.output_fields}

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

    def initialise_reader(self, data_param, task_param):
        """
        task_param specifies how to combine user input modalities
        e.g., for multimodal segmentation 'image' corresponds to multiple
        modality sections, 'label' corresponds to one modality section
        """
        app_type = task_param['name']
        self._file_list = VolumeReader._load_and_merge_csv_files(data_param)

        if app_type == "net_segmentation.py":
            from niftynet.application.segmentation_application \
                import SUPPORTED_INPUT
            # only choose fields that are supported by the application
            # (SUPPORTED_INPUT) and have user parameter specification
        elif app_type == "net_gan.py":
            from niftynet.application.gan_application \
                import SUPPORTED_INPUT

        if not self.output_fields:
            self.output_fields = SUPPORTED_INPUT
        self.output_fields = [field_name
                              for field_name in task_param
                              if field_name in self.output_fields and
                              task_param[field_name] != ()]
        self._input_sources = {field: task_param[field]
                               for field in self.output_fields}
        self.output_list = self._filename_to_image_list(data_param)
        self.current_id = -1

        tf.logging.info('initialised reader: loading {} from {}'.format(
            self.output_fields, self.input_sources))

    def layer_op(self, shuffle=True):
        """
        this layer returns a dictionary
          keys: self.output_fields
          values: image volume objects
        """
        if shuffle:
            # this is thread safe, don't access self.current_id
            idx = np.random.randint(len(self.output_list))
            return self.output_list[idx]
        else:
            # this is not thread safe
            idx = self.current_id + 1
            self.current_id = idx

            if idx < len(self.output_list) and idx >= 0:
                return self.output_list[idx]
            else:
                # return nothing if current_id is not valid
                return None

    def _filename_to_image_list(self, data_param):
        """
        converting a list of filenames to a list of image objects
        useful properties (e.g. interp_order) are added to each object
        """
        volume_list = []
        for row_id in range(len(self._file_list)):
            # combine fieldnames and volumes as a dictionary
            _dict = {field: self._create_image(row_id,
                                               self.input_sources[field],
                                               data_param)
                     for field in self.input_sources}
            volume_list.append(_dict)
        return volume_list

    def _create_image(self, idx, modalities, data_param):
        """
        data_param consists of discription of each modality
        This function combines modalities according to the 'modalities'
        parameter and create <niftynet.io.input_type.SpatialImage*D>
        """
        try:
            file_path = tuple([self._file_list.loc[idx, mod]
                               for mod in modalities])
            interp_order = tuple([data_param[mod].interp_order
                                  for mod in modalities])
            output_pixdim = tuple([data_param[mod].pixdim
                                   for mod in modalities])
            output_axcodes = tuple([data_param[mod].axcodes
                                   for mod in modalities])
        except KeyError:
            tf.logging.fatal(
                "Specified modality names {} "
                "not found in config: input sections {}".format(
                    modalities, list(data_param)))
            raise
        image_properties = {'file_path': file_path,
                            'name': modalities,
                            'interp_order': interp_order,
                            'output_pixdim': output_pixdim,
                            'output_axcodes': output_axcodes}
        image = ImageFactory.create_instance(**image_properties)
        return image

    @staticmethod
    def _load_and_merge_csv_files(data_param):
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
