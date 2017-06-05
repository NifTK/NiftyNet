# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from random import shuffle

import numpy as np

from layer.base_layer import Layer


class VolumeLoaderLayer(Layer):
    """
    This class manages the loading step, i.e., return subject's data
    by searching user provided path and modality constraints.
    The volumes are resampled/reoriented if required.

    This class maintains a list of subjects, where each element of the list
    is a Patient object.
    """

    def __init__(self,
                 csv_reader,
                 standardisor=None,
                 is_training=True,
                 do_reorientation=True,
                 do_resampling=True,
                 spatial_padding=None,
                 do_normalisation=True,
                 do_whitening=True,
                 interp_order=(3, 0),
                 name='volume_loader'):

        super(VolumeLoaderLayer, self).__init__(name=name)

        self.csv_table = csv_reader
        self.standardisor = standardisor

        self.is_training = is_training
        self.do_reorientation = do_reorientation
        self.do_resampling = do_resampling
        self.spatial_padding = spatial_padding
        self.do_normalisation = do_normalisation
        self.do_whitening = do_whitening
        self.interp_order = interp_order

        self.columns_to_load = (0, 1, 2)
        while len(self.interp_order) < len(self.columns_to_load):
            self.interp_order = self.interp_order + (3,)
        self.interp_order = self.interp_order[:len(self.columns_to_load)]

        self.subject_list = None
        self.current_id = None
        self.__initialise_subject_list()

    def __initialise_subject_list(self):
        """
        provide a list of subjects, the subjects are constructed from csv_table
        data. These are used to train a histogram normalisation reference.
        """
        self.subject_list = self.csv_table.to_subject_list()
        if len(self.subject_list) == 0:
            raise ValueError("subject not found")
        if self.is_training:
            shuffle(self.subject_list)
        self.current_id = -1

        if self.standardisor is None:
            return
        standardisor_ready = self.standardisor.is_ready(self.subject_list,
                                                        self.do_normalisation,
                                                        self.do_whitening)
        if not standardisor_ready:
            if self.is_training:
                self.standardisor.train_normalisation_ref(self.subject_list)
            else:
                raise ValueError(
                    "histogram normalisation enabled, but can't find histogram"
                    "reference model file")
        else:
            print('histogram normalisation initialised')

    def layer_op(self):
        """
        Call this function to get the next subject's image data.
        """
        # go to the next subject in the list (avoid running out of the list)
        if self.is_training:
            idx = np.random.randint(len(self.subject_list))
        else:
            idx = self.current_id + 1
            self.current_id = idx  # warning this is not thread-safe
        current_subject = self.subject_list[idx]
        # print current_subject
        subject_dict = current_subject.load_columns(self.columns_to_load,
                                                    self.do_reorientation,
                                                    self.do_resampling,
                                                    self.interp_order,
                                                    self.spatial_padding)
        image = subject_dict['input_image_file']
        label = subject_dict['target_image_file']
        weight = subject_dict['weight_map_file']

        if self.standardisor is not None:
            image.data = self.standardisor(image.data,
                                           self.do_normalisation,
                                           self.do_whitening)
        return image, label, weight, idx

    @property
    def has_next(self):
        if self.is_training:
            return True
        if self.current_id < len(self.subject_list) - 1:
            return True
        print('volume loader finished (reaching the last element)')
        return False

    @property
    def num_subject(self):
        return len(self.subject_list)

    def num_modality(self, column_id):
        column_i = self.subject_list[0].column(column_id)
        if column_i is None:
            return 0
        num_modality = column_i.num_modality
        # make sure num_modality is the same across the list of subjects
        for x in self.subject_list:
            assert x.column(column_id).num_modality == num_modality
        return column_i.num_modality

    def get_subject(self, idx):
        return self.subject_list[idx]
