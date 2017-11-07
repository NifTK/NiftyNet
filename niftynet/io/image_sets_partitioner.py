# -*- coding: utf-8 -*-
"""
This module manages a table of subject ids and
their associated image file names.
A subset of the table can be retrieved by partitioning the set of images into
subsets of `train`, `validation`, `inference`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random

import pandas
import tensorflow as tf  # to use the system level logging

from niftynet.utilities.decorators import singleton
from niftynet.utilities.filename_matching import KeywordsMatching
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig
from niftynet.utilities.util_common import look_up_operations
from niftynet.utilities.util_csv import match_and_write_filenames_to_csv
from niftynet.utilities.util_csv import write_csv

COLUMN_UNIQ_ID = 'subject_id'
TRAIN_ID = 'train'
VALID_ID = 'validation'
INFER_ID = 'inference'
SUPPORTED_PHASES = {TRAIN_ID, VALID_ID, INFER_ID}


@singleton
class ImageSetsPartitioner(object):
    def __init__(self,
                 data_param,
                 dataset_split_file="",
                 ratios=(0.1, 0.1)):
        self.data_param = data_param
        self.ratios = ratios
        self.dataset_split_file = dataset_split_file
        self.default_image_file_location = \
            NiftyNetGlobalConfig().get_niftynet_home_folder()
        self._file_list = None

    def initialise(self):
        self.load_data_sections_by_subject()
        self.split_dataset()

    @property
    def number_of_subjects(self):
        return self._file_list.shape[0].count()

    def get_file_list(self, phase='train'):
        phase = look_up_operations(phase.lower(), SUPPORTED_PHASES)
        pass

    def load_data_sections_by_subject(self):
        if not self.data_param:
            tf.logging.fatal(
                'Nothing to load, please check input sections in the config.')
            raise ValueError
        self._file_list = None
        for section_name in self.data_param:
            modality_file_list = self.grep_files_by_data_section(section_name)
            if self._file_list is None:
                # adding all rows of the first modality
                self._file_list = modality_file_list
                continue
            n_rows = self._file_list.shape[0].count()
            self._file_list = pandas.merge(self._file_list,
                                           modality_file_list,
                                           on=COLUMN_UNIQ_ID)
            if self._file_list.shape[0].count() < n_rows:
                tf.logging.warning('rows not matched in section [%s]',
                                   section_name)

        if self._file_list is None or self._file_list.size == 0:
            tf.logging.fatal(
                "empty filename lists, please check the csv "
                "files. (removing csv_file keyword if it is in the config file "
                "to automatically search folders and generate new csv "
                "files again)\n\n"
                "Please note in the matched file names, each subject id are "
                "created by removing all keywords listed `filename_contains` "
                "in the config.\n\n"
                "E.g., `filename_contains=foo, bar` will match file "
                "foo_subject42_bar.nii.gz, and the subject id is _subject42_.")
            raise IOError

    def grep_files_by_data_section(self, modality_name):
        """
        list all files by a given input data section,
        if the `csv_file` property of the section corresponds to a file,
            read the list from the file;
        otherwise
            write the list to `csv_file`.

        returns: a table with two columns,
                 the column names are (COLUMN_UNIQ_ID, modality_name)
        """
        if modality_name not in self.data_param:
            tf.logging.fatal('unknown section name [%s], '
                             'current input section names: %s.',
                             modality_name, list(self.data_param))
            raise ValueError

        # input data section must have a `csv_file` section for loading
        # or writing filename lists
        try:
            csv_file = self.data_param[modality_name].csv_file
        except AttributeError:
            tf.logging.fatal('Missing `csv_file` field in the config file, '
                             'unknown configuration format.')
            raise

        if hasattr(self.data_param[modality_name], 'path_to_search') and \
                len(self.data_param[modality_name].path_to_search):
            tf.logging.info('[%s] search file folders, writing csv file %s',
                            modality_name, csv_file)
            section_properties = self.data_param[modality_name].__dict__.items()
            # grep files by section properties and write csv
            matcher = KeywordsMatching.from_tuple(
                section_properties,
                self.default_image_file_location)
            match_and_write_filenames_to_csv([matcher], csv_file)
        else:
            tf.logging.info(
                '[%s] using existing csv file %s, skipped filenames search',
                modality_name, csv_file)

        if not os.path.isfile(csv_file):
            tf.logging.fatal('[%s] csv file %s not found.',
                             modality_name, csv_file)
            raise IOError
        try:
            csv_list = pandas.read_csv(
                csv_file, header=None, names=[COLUMN_UNIQ_ID, modality_name])
        except Exception as e:
            tf.logging.fatal(repr(e))
            raise
        return csv_list

    def split_dataset(self):
        validation_fraction, inference_fraction = self.ratios
        count_valid = math.ceil(self.number_of_subjects * validation_fraction)
        count_infer = math.ceil(self.number_of_subjects * inference_fraction)
        count_train = self.number_of_subjects - count_infer - count_valid
        phases = [TRAIN_ID] * count_train + \
                 [VALID_ID] * count_valid + \
                 [INFER_ID] * count_infer
        random.shuffle(phases)
        write_csv(self.dataset_split_file,
                  zip(self._file_list[COLUMN_UNIQ_ID], phases))
