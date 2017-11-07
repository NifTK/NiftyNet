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
COLUMN_PHASE = 'phase'
TRAIN_ID = 'train'
VALID_ID = 'validation'
INFER_ID = 'inference'
ALL_ID = 'all'
SUPPORTED_PHASES = {TRAIN_ID, VALID_ID, INFER_ID, ALL_ID}


@singleton
class ImageSetsPartitioner(object):
    _file_list = None
    _partition_ids = None
    data_param = None
    ratios = None

    # for saving the splitting index
    dataset_split_file = ""
    # default location for searching the image files
    default_image_file_location = \
        NiftyNetGlobalConfig().get_niftynet_home_folder()

    def initialise(self,
                   data_param,
                   new_partition=True,
                   data_split_file="./test.csv",
                   ratios=(0.1, 0.1)):
        self.data_param = data_param
        self.dataset_split_file = data_split_file
        self.ratios = ratios
        self.load_data_sections_by_subject()
        self.randomly_split_dataset(overwrite=new_partition)

    def number_of_subjects(self, phase='all'):
        phase = look_up_operations(phase.lower(), SUPPORTED_PHASES)
        if phase == ALL_ID:
            return self._file_list[COLUMN_UNIQ_ID].count()
        selector = self._partition_ids[COLUMN_PHASE] == phase
        return self._partition_ids[selector].count()[COLUMN_UNIQ_ID]

    def get_file_list(self, phase='train'):
        phase = look_up_operations(phase.lower(), SUPPORTED_PHASES)
        if phase == ALL_ID:
            self._file_list = self._file_list.sort_index()
            return self._file_list
        selector = self._partition_ids[COLUMN_PHASE] == phase
        selected = self._partition_ids[selector][[COLUMN_UNIQ_ID]]
        if selected.empty:
            return []
        return pandas.merge(self._file_list, selected, on=COLUMN_UNIQ_ID)

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
            n_rows = self._file_list[COLUMN_UNIQ_ID].count()
            self._file_list = pandas.merge(self._file_list,
                                           modality_file_list,
                                           on=COLUMN_UNIQ_ID)
            if self._file_list[COLUMN_UNIQ_ID].count() < n_rows:
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

    def randomly_split_dataset(self, overwrite=False):
        if overwrite:
            validation_fraction, inference_fraction = self.ratios
            n_total = self.number_of_subjects()
            n_valid = int(math.ceil(n_total * validation_fraction))
            n_infer = int(math.ceil(n_total * inference_fraction))
            n_train = int(n_total - n_infer - n_valid)
            phases = [TRAIN_ID] * n_train + \
                     [VALID_ID] * n_valid + \
                     [INFER_ID] * n_infer
            random.shuffle(phases)
            write_csv(self.dataset_split_file,
                      zip(self._file_list[COLUMN_UNIQ_ID], phases))

        if os.path.isfile(self.dataset_split_file):
            self._partition_ids = pandas.read_csv(
                self.dataset_split_file,
                header=None,
                names=[COLUMN_UNIQ_ID, COLUMN_PHASE])
            return
        tf.logging.fatal('Unable to load partition file from %s',
                         self.dataset_split_file)
        raise ValueError

    def __str__(self):
        return self.to_string()

    def to_string(self):
        summary_str = '\nNumber of subjects {}, '.format(
            self.number_of_subjects())
        if self.ratios:
            summary_str += 'data partitioning (percentage): \n' \
                           '-- {} {} ({}),\n' \
                           '-- {} {} ({}), \n' \
                           '-- {} {}.\n'.format(
                VALID_ID, self.number_of_subjects(VALID_ID), self.ratios[0],
                INFER_ID, self.number_of_subjects(INFER_ID), self.ratios[1],
                TRAIN_ID, self.number_of_subjects(TRAIN_ID))
        else:
            summary_str += '-- using all subjects ' \
                           '(without data partitioning).\n'
        return summary_str
