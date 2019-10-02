# -*- coding: utf-8 -*-
"""
This module manages a table of subject ids and
their associated image file names.
A subset of the table can be retrieved by partitioning the set of images into
subsets of ``Train``, ``Validation``, ``Inference``.
"""
from __future__ import absolute_import, division, print_function

import os
import shutil

import pandas
import tensorflow as tf  # to use the system level logging

from niftynet.io.base_sets_partitioner import (
    COLUMN_UNIQ_ID, DEFAULT_SPLIT_FILE_NAME, BaseSetsPartitioner)
from niftynet.utilities.filename_matching import KeywordsMatching
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig
from niftynet.utilities.util_csv import match_and_write_filenames_to_csv


class ImageSetsPartitioner(BaseSetsPartitioner):
    """
    This class maintains a pandas.dataframe of filenames for all input sections
    The list of filenames are obtained by searching the specified folders
    or loading from an existing csv file.

    Users can query a subset of the dataframe by train/valid/infer partition
    label and input section names.
    """

    def __init__(self, data_param, data_split_file=None):
        """

        :param data_param: input config sections
        """
        data_split_file = data_split_file or DEFAULT_SPLIT_FILE_NAME
        default_folder = os.path.dirname(data_split_file)
        file_list = load_data_sections_by_subject(data_param, default_folder)

        BaseSetsPartitioner.__init__(
            self, file_list=file_list, data_split_file=data_split_file)


def load_data_sections_by_subject(data_param, default_folder):
    """
    Go through all input data sections, converting each section
    to a list of file names.

    These lists are merged on ``COLUMN_UNIQ_ID``.

    This function returns ``file_list``.
    """
    if not data_param:
        tf.logging.fatal(
            'Nothing to load, please check input sections in the config.')
        raise ValueError
    file_list = None
    for section_name in data_param:
        modality_file_list = grep_files_by_data_section(
            section_name, data_param, default_folder)
        if file_list is None:
            # adding all rows of the first modality
            file_list = modality_file_list
            continue
        n_rows = file_list[COLUMN_UNIQ_ID].count()
        file_list = pandas.merge(
            file_list, modality_file_list, how='outer', on=COLUMN_UNIQ_ID)
        if file_list[COLUMN_UNIQ_ID].count() < n_rows:
            tf.logging.warning('rows not matched in section [%s]',
                               section_name)

    if file_list is None or file_list.size == 0:
        tf.logging.fatal(
            "Empty filename lists, please check the csv "
            "files (removing csv_file keyword if it is in the config file "
            "to automatically search folders and generate new csv "
            "files again).\n\n"
            "Please note in the matched file names, each subject id are "
            "created by removing all keywords listed `filename_contains` "
            "in the config.\n"
            "E.g., `filename_contains=foo, bar` will match file "
            "foo_subject42_bar.nii.gz, and the subject id is "
            "_subject42_.\n\n")
        raise IOError
    return file_list


def grep_files_by_data_section(modality_name, data_param, default_csv_folder):
    """
    list all files by a given input data section::
        if the ``csv_file`` property of ``data_param[modality_name]``
        corresponds to a file, read the list from the file;
        otherwise
            write the list to ``csv_file``.

    :return: a table with two columns,
             the column names are ``(COLUMN_UNIQ_ID, modality_name)``.
    """
    if modality_name not in data_param:
        tf.logging.fatal(
            'unknown section name [%s], '
            'current input section names: %s.', modality_name,
            list(data_param))
        raise ValueError

    # input data section must have a ``csv_file`` section for loading
    # or writing filename lists
    if isinstance(data_param[modality_name], dict):
        mod_spec = data_param[modality_name]
    else:
        mod_spec = vars(data_param[modality_name])

    #########################
    # guess the csv_file path
    #########################
    temp_csv_file = None
    try:
        csv_file = os.path.expanduser(mod_spec.get('csv_file', None))
        if not os.path.isfile(csv_file):
            # writing to the same folder as data_split_file
            default_csv_file = os.path.join(default_csv_folder,
                                            '{}.csv'.format(modality_name))
            tf.logging.info(
                '`csv_file = %s` not found, '
                'writing to "%s" instead.', csv_file, default_csv_file)
            csv_file = default_csv_file
            if os.path.isfile(csv_file):
                tf.logging.info('Overwriting existing: "%s".', csv_file)
        csv_file = os.path.abspath(csv_file)
    except (AttributeError, KeyError, TypeError):
        tf.logging.debug('`csv_file` not specified, writing the list of '
                         'filenames to a temporary file.')
        import tempfile
        temp_csv_file = os.path.join(tempfile.mkdtemp(),
                                     '{}.csv'.format(modality_name))
        csv_file = temp_csv_file

    def _clear_tmp():
        if temp_csv_file:
            shutil.rmtree(os.path.dirname(temp_csv_file), ignore_errors=True)

    ##############################################
    # writing csv file if path_to_search specified
    ##############################################
    if mod_spec.get('path_to_search', None):
        tf.logging.info('[%s] search file folders, writing csv file %s',
                        modality_name,
                        csv_file if not temp_csv_file else 'temp')
        # grep files by section properties and write csv
        try:
            matcher = KeywordsMatching.from_dict(
                input_dict=mod_spec,
                default_folder=NiftyNetGlobalConfig().get_niftynet_home_folder(
                ))
            match_and_write_filenames_to_csv([matcher], csv_file)
        except (IOError, ValueError) as reading_error:
            tf.logging.warning(
                'Ignoring input section: [%s], '
                'due to the following error:', modality_name)
            tf.logging.warning(repr(reading_error))
            _clear_tmp()
            return pandas.DataFrame(columns=[COLUMN_UNIQ_ID, modality_name])
    else:
        tf.logging.info(
            '[%s] using existing csv file %s, skipped filenames search',
            modality_name, csv_file)

    #################################
    # loading the file as a dataframe
    #################################
    try:
        return pandas.read_csv(
            csv_file,
            header=None,
            dtype=(str, str),
            names=[COLUMN_UNIQ_ID, modality_name],
            skipinitialspace=True)
    except Exception as csv_error:
        tf.logging.fatal(repr(csv_error))
        _clear_tmp()
        raise
