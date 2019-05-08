# -*- coding: utf-8 -*-
"""
This module manages a table of subject ids and
their associated image entries.
A subset of the table can be retrieved by partitioning the set of images into
subsets of ``Train``, ``Validation``, ``Inference``.
"""

from abc import ABCMeta, abstractmethod, abstractproperty
import math
import random

import tensorflow as tf

from niftynet.engine.signal import TRAIN, VALID, INFER, ALL
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_PHASES = {TRAIN, VALID, INFER, ALL}
COLUMN_UNIQ_ID = 'subject_id'
COLUMN_PHASE = 'phase'

class BaseImageSetsPartitioner(object):
    """
    Base class for image sets partitioners
    """

    __metaclass__ = ABCMeta

    ratios = None
    data_param = None

    # pylint: disable=unused-argument
    def initialise(self,
                   data_param,
                   new_partition=False,
                   data_split_file=None,
                   ratios=None):
        """
        Set the data partitioner parameters

        :param new_partition: bool value indicating whether to generate new
            partition ids and overwrite csv file
            (this class will write partition file iff new_partition)
        :param data_split_file: location of the partition id file
        :param ratios: a tuple/list with two elements:
            ``(fraction of the validation set, fraction of the inference set)``
            initialise to None will disable data partitioning
            and get_file_list always returns all subjects.
        """

        self.ratios = ratios
        self.data_param = data_param

    @abstractmethod
    def number_of_subjects(self, phase=ALL):
        """
        query number of images according to phase.

        :param phase: application phase; TRAIN, VALID, etc.
        :return: number of subjects in partition
        """

        return

    def _create_partitions(self):
        """
        Creates the subject partitions as linear array of application
        phases, where each entry corresponds to a subject.
        :return: a list of phases with one entry for every subject in
        the data set.
        """

        if self.number_of_subjects() <= 0:
            raise RuntimeError('Called on an uninitialised partitioner.')

        try:
            valid_fraction, infer_fraction = self.ratios
            valid_fraction = max(min(1.0, float(valid_fraction)), 0.0)
            infer_fraction = max(min(1.0, float(infer_fraction)), 0.0)
        except (TypeError, ValueError):
            tf.logging.fatal(
                'Unknown format of faction values %s', self.ratios)
            raise

        if (valid_fraction + infer_fraction) <= 0:
            tf.logging.warning(
                'To split dataset into training/validation, '
                'please make sure '
                '"exclude_fraction_for_validation" parameter is set to '
                'a float in between 0 and 1. Current value: %s.',
                valid_fraction)
            # raise ValueError

        n_total = self.number_of_subjects()
        n_valid = int(math.ceil(n_total * valid_fraction))
        n_infer = int(math.ceil(n_total * infer_fraction))
        n_train = int(n_total - n_infer - n_valid)
        phases = [TRAIN] * n_train + [VALID] * n_valid + [INFER] * n_infer
        if len(phases) > n_total:
            phases = phases[:n_total]
        random.shuffle(phases)

        return phases

    def _look_up_phase(self, phase):
        """
        :return: the phase in canonical form
        """

        try:
            return look_up_operations(phase.lower(), SUPPORTED_PHASES)
        except (ValueError, AttributeError):
            tf.logging.fatal('Unknown phase argument.')
            raise

    @abstractmethod
    def get_image_lists_by(self, phase=None, action='train'):
        """
        Get file lists by action and phase.

        This function returns file lists for training/validation/inference
        based on the phase or action specified by the user.

        ``phase`` has a higher priority:
        If `phase` specified, the function returns the corresponding
        file list (as a list).

        otherwise, the function checks ``action``:
        it returns train and validation file lists if it's training action,
        otherwise returns inference file list.

        :param action: an action
        :param phase: an element from ``{TRAIN, VALID, INFER, ALL}``
        :return:
        """

        return

    @abstractmethod
    def has_phase(self, phase):
        """

        :return: True if the `phase` subset of images is not empty.
        """

        return

    @property
    def has_training(self):
        """

        :return: True if the TRAIN subset of images is not empty.
        """
        return self.has_phase(TRAIN)

    @property
    def has_inference(self):
        """

        :return: True if the INFER subset of images is not empty.
        """
        return self.has_phase(INFER)

    @property
    def has_validation(self):
        """

        :return: True if the VALID subset of images is not empty.
        """
        return self.has_phase(VALID)

    @abstractproperty
    def all_files(self):
        """

        :return: list of all filenames
        """

        return

    def reset(self):
        """
        reset all fields of this singleton class.
        """
