# -*- coding: utf-8 -*-
"""
This module manages a table of subject ids and
their associated image ids.

A subset of the table can be retrieved by partitioning the set of images into
subsets of ``Train``, ``Validation``, ``Inference``.
"""
from __future__ import absolute_import, division, print_function

import pandas

from niftynet.io.base_sets_partitioner import (COLUMN_UNIQ_ID,
                                               BaseSetsPartitioner)


class RangeSetsPartitioner(BaseSetsPartitioner):
    """
    This class maintains a pandas.dataframe of subjects for all
    modality names

    Users can query a subset of the dataframe by train/valid/infer partition
    label and input section names.
    """

    def __init__(self, num_subjects, modality_names, data_split_file=None):
        """

        :param num_subjects: integer -- total number of subjects
        :param modality_names: columns of the file_list
        """
        file_list = _generate_file_list(num_subjects, modality_names)

        BaseSetsPartitioner.__init__(
            self, file_list=file_list, data_split_file=data_split_file)


def _generate_file_list(num_subjects, modality_names):
    subjects = list(range(num_subjects))
    data_dict = {COLUMN_UNIQ_ID: subjects}
    for mod in modality_names:
        data_dict[mod] = subjects
    return pandas.DataFrame(data=data_dict)
