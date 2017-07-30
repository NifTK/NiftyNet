# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from niftynet.layer.base_layer import Layer
from niftynet.io.input_type import ImageFactory
from niftynet.utilities.user_parameters_helper import NULL_STRING_TUPLE

import pandas
import os
import tensorflow as tf
import numpy as np

class VolumeReader(Layer):

    def __init__(self):
        self.file_list = None
        self.output_fields = None
        self.outputs = None
        self.current_id = None
        super(VolumeReader, self).__init__(name='volume_reader')

    def initialise_reader(self, data_param, task_param):
        """
        task_param specifies how to combine user input modalities
        e.g., for multimodal segmentation 'image' corresponds to multiple
        modality sections, 'label' corresponds to one modality section
        """
        app_type = task_param.name
        self.file_list = self._load_file_list(data_param)

        if app_type == "net_segmentation.py":
            from niftynet.application.segmentation_application\
                import SUPPORTED_INPUT
            self.output_fields = [field for field in vars(task_param)
                                  if field in SUPPORTED_INPUT and
                                  vars(task_param)[field] != NULL_STRING_TUPLE]
        self.outputs = self.filename_to_image_list(data_param, task_param)
        self.current_id = -1

    def filename_to_image_list(self, data_param, task_param):
        volume_list = []
        for idx in range(len(self.file_list)):
            # initialise a reader output dictionary
            subject_dict = dict(zip(self.output_fields,
                                    (None,)*len(self.output_fields)))
            for input_name in self.output_fields:
                modalities = tuple(vars(task_param)[input_name])
                subject_dict[input_name] = self.__create_volume(
                        idx, modalities, data_param)
            volume_list.append(subject_dict)
        return volume_list

    def layer_op(self, is_training=False):
        if is_training:
            idx = np.random.randint(len(self.outputs))
        else:
            # this is not thread safe
            idx = self.current_id + 1
            self.current_id = idx
        return self.outputs[idx]

    def __create_volume(self, idx, modalities, data_param):
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

        image = ImageFactory.create_instance(
            file_path=file_path,
            name=modalities,
            interp_order=interp_order)
        return image


    def _load_file_list(self, data_param):
        """
        Converts a list of csv_files in data_param
        in to a joint list (using extact matching)
        This function returns a pandas.core.frame.DataFrame of joint list
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
