# -*- coding: utf-8 -*-
"""
This module provides image set partitioning for
images kept in memory.
"""
from __future__ import absolute_import

import os

import pandas

from niftynet.engine.signal import ALL, INFER, TRAIN, VALID
from niftynet.io.image_sets_partitioner import (
    COLUMN_PHASE, COLUMN_UNIQ_ID, SUPPORTED_PHASES, BaseImageSetsPartitioner)
from niftynet.utilities.decorators import singleton

# Name of the data_param namespace entry for number of in-memory subjects
MEMORY_INPUT_NUM_SUBJECTS_PARAM = 'num_subjects'


@singleton
class MemoryImageSetsPartitioner(BaseImageSetsPartitioner):
    """
    Partitioning of images kept in RAM.
    """

    _partitions = None
    _num_subjects = None

    def initialise(self,
                   data_param,
                   new_partition=False,
                   data_split_file=None,
                   ratios=None):
        """
        :param data_param: data specification including the number of subjects
            to partition.
        :param new_partition: bool value indicating whether to generate new
            partition ids and overwrite csv file
            (this class will write partition file iff new_partition)
        :param data_split_file: location of the partition id file
        :param ratios: a tuple/list with two elements:
            ``(fraction of the validation set, fraction of the inference set)``
            initialise to None will disable data partitioning
            and get_file_list always returns all subjects.
        """

        self._num_subjects = data_param[MEMORY_INPUT_NUM_SUBJECTS_PARAM]
        self.ratios = ratios

        if new_partition or not data_split_file \
           or not os.path.isfile(data_split_file):
            self._partitions = self._assemble_partitions()
            if data_split_file:
                self._write_partions(data_split_file)

        else:
            assert os.path.isfile(data_split_file)
            self._partitions = self._load_partitions(data_split_file)

    def number_of_subjects(self, phase=ALL):
        phase = self._look_up_phase(phase)

        return self._num_subjects if phase == ALL \
            else len(self._partitions[phase])

    def _assemble_partitions(self):
        """
        Builds a new partition table dividing the subjects
        between the various phases according to the specified ratios
        """

        phases = self._create_partitions()

        partitions = {phase: [] for phase in SUPPORTED_PHASES}
        for idx, phase in enumerate(phases):
            partitions[phase].append(idx)

        return partitions

    def _write_partions(self, path):
        """
        Persistently saves the current partioning in the FS.
        :param path: destination path for the CSV file.
        """

        phases = []
        indices = []
        for phase, idcs in self._partitions.items():
            phases += [phase] * len(idcs)
            indices += idcs

        data_frame = pandas.DataFrame({
            COLUMN_PHASE: phases,
            COLUMN_UNIQ_ID: indices
        })
        data_frame = data_frame.sort_values(by=[COLUMN_UNIQ_ID])
        data_frame.to_csv(path)

    @staticmethod
    def _load_partitions(path):
        """
        Loads a partitioning that was previously written to disc.
        :param path: path to the CSV file containing the data partions
        """

        data_frame = pandas.read_csv(path)

        if len(data_frame.columns) != 2 \
           or COLUMN_PHASE not in data_frame.columns \
           or COLUMN_UNIQ_ID not in data_frame.columns:
            raise ValueError(
                '{} is not a memory data-set partitions file;'.format(path) +
                ' have you accidentally loaded a F/S-based one?')

        partitions = {}
        for phase in SUPPORTED_PHASES:
            partitions[phase] = data_frame.loc[data_frame[COLUMN_PHASE] ==
                                               phase,
                                               COLUMN_UNIQ_ID].values.tolist()

        return partitions

    def get_image_lists_by(self, phase=None, action='train'):
        lists = []

        if phase:
            if phase == ALL:
                lists = [idcs for idcs in self._partitions.values()]
            else:
                lists = [self._partitions[phase]]
        elif action and TRAIN.startswith(action):
            lists = [self._partitions[TRAIN], self._partitions[VALID]]
        else:
            lists = [self._partitions[INFER]]

        return lists

    def has_phase(self, phase):
        return phase in self._partitions and self._partitions[phase]

    def all_files(self):
        return list(range(self._num_subjects))

    def reset(self):
        super(MemoryImageSetsPartitioner, self).reset()

        self._partitions = None
        self._num_subjects = 0


def set_number_of_memory_subjects(data_param, num_subjects):
    """
    Configures the data parameters with the number of subjects
    expected to be retrievable via image input callback
    functions.
    :param data_param: the data specification dictionary for
        the application
    :param num_subjects: the number of subjects to expect.
    :return: the modified data_param dictionary
    """

    data_param[MEMORY_INPUT_NUM_SUBJECTS_PARAM] = num_subjects

    return data_param


def restore_data_param(data_param):
    """
    Clears any potentially conflicting settings from the
    data_param dictionary. Must be called after using
    set_number_of_memory_subjects and before starting
    the application.
    :param data_param: a modified data specification dictionary
    :return: the restored dictionary
    """

    del data_param[MEMORY_INPUT_NUM_SUBJECTS_PARAM]

    return data_param


def is_memory_data_param(data_param):
    """
    Indicates whether the argument data-specification dictionary
    is one where the images are expected to be retrieved using
    input callback functions.
    """

    return (MEMORY_INPUT_NUM_SUBJECTS_PARAM in data_param
            and data_param[MEMORY_INPUT_NUM_SUBJECTS_PARAM] > 0)
