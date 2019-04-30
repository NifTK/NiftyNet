# -*- coding: utf-8 -*-
"""
This module provides image set partitioning for
images kept in memory.
"""
from __future__ import absolute_import

import os

import pandas

from niftynet.engine.signal import TRAIN, VALID, INFER, ALL
from niftynet.io.image_sets_partitioner import BaseImageSetsPartitioner,\
    SUPPORTED_PHASES,\
    COLUMN_UNIQ_ID,\
    COLUMN_PHASE
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
        :param num_subjects: number of subjects to partition
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
            phases += [phase]*len(idcs)
            indices += idcs

        df = pandas.DataFrame({COLUMN_PHASE: phases, COLUMN_UNIQ_ID: indices})
        df = df.sort_values(by=[COLUMN_UNIQ_ID])
        df.to_csv(path)

    def _load_partitions(self, path):
        """
        Loads a partitioning that was previously written to disc.
        :param path: path to the CSV file containing the data partions
        """

        df = pandas.DataFrame.from_csv(path)

        partitions = {}
        for phase in SUPPORTED_PHASES:
            partitions[phase] = df.loc(df[COLUMN_PHASE]==phase, COLUMN_UNIQ_ID)\
                                  .values.tolist()

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
        return phase in self._partitions and not self._partitions[phase]

    def reset(self):
        super(MemoryImageSetsPartitioner, self).reset()

        self._partitions = None
        self._num_subjects = 0
