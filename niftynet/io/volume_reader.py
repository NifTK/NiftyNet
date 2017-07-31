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
from niftynet.utilities.user_parameters_helper import validate_input_tuple


class VolumeReader(Layer):
    def __init__(self, output_fields):
        self.output_fields = output_fields
        self.file_list = None
        self.output_list = None
        self.current_id = None
        super(VolumeReader, self).__init__(name='volume_reader')

    @property
    def output_fields(self):
        return self._output_fields

    @output_fields.setter
    def output_fields(self, fields_tuple):
        # output_fields is a sequence of output names
        # each name might correspond to a list of multiple input sources
        # this should be specified in CUSTOM section in the config
        self._output_fields = validate_input_tuple(fields_tuple, basestring)

    def initialise_reader(self, data_param, task_param):
        """
        task_param specifies how to combine user input modalities
        e.g., for multimodal segmentation 'image' corresponds to multiple
        modality sections, 'label' corresponds to one modality section
        """
        app_type = task_param['name']
        self.file_list = VolumeReader._load_and_merge_csv_files(data_param)

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
        self.output_list = self._filename_to_image_list(data_param, task_param)
        self.current_id = -1

    def layer_op(self, is_training=False):
        """
        this layer returns a dictionary
          keys: self.output_fields
          values: image volume objects
        """
        if is_training:
            # this is thread safe, don't access self.current_id
            idx = np.random.randint(len(self.output_list))
        else:
            # this is not thread safe
            idx = self.current_id + 1
            self.current_id = idx
        return self.output_list[idx]

    def _filename_to_image_list(self, data_param, task_param):
        """
        converting a list of filenames to a list of image objects
        useful properties (e.g. interp_order) are added to each object
        """
        volume_list = []
        for row_id in range(len(self.file_list)):
            # combine fieldnames and volumes as a dictionary
            volume_dict = {field: self._create_image(row_id,
                                                     task_param[field],
                                                     data_param)
                           for field in self.output_fields}
            volume_list.append(volume_dict)
        return volume_list

    def _create_image(self, idx, modalities, data_param):
        """
        data_param consists of discription of each modality
        This function combines modalities according to the 'modalities'
        parameter and create <niftynet.io.input_type.SpatialImage*D>
        """
        try:
            file_path = tuple([self.file_list.loc[idx, mod]
                               for mod in modalities])
            interp_order = tuple([data_param[mod].interp_order
                                  for mod in modalities])
        except KeyError:
            tf.logging.fatal(
                "Specified modality names {} "
                "not found in config: input sections {}".format(
                    modalities, list(data_param)))
            raise
        image_properties = {'file_path': file_path,
                            'name': modalities,
                            'interp_order': interp_order}
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
        file_list = None
        for modality_name in data_param:
            csv_file = data_param.get(modality_name, '').csv_file
            if not os.path.isfile(csv_file):
                tf.logging.fatal("csv file {} not found.".format(csv_file))
                raise IOError
            csv_list = pandas.read_csv(
                csv_file, header=None, names=['subject_id', modality_name])
            if file_list is None:
                file_list = csv_list
                continue
            # merge file_list based on subject_ids (first column of each csv)
            n_rows = file_list.shape[0]
            file_list = pandas.merge(file_list, csv_list, on='subject_id')
            if file_list.shape[0] != n_rows:
                tf.logging.warning("rows not matched in {}".format(csv_file))
        if file_list.size == 0:
            tf.logging.fatal("no common subject_ids in filename lists,"
                             "please check the csv files.")
            raise IOError
        return file_list
