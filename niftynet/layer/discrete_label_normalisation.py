# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import os

import numpy as np
import tensorflow as tf

import niftynet.utilities.histogram_standardisation as hs
from niftynet.layer.base_layer import DataDependentLayer
from niftynet.utilities.misc_common import printProgressBar
from niftynet.utilities.user_parameters_helper import standardise_string


class DiscreteLabelNormalisationLayer(DataDependentLayer):
    def __init__(self,
                 field,
                 modalities,
                 model_filename,
                 name='label_norm'):

        super(DiscreteLabelNormalisationLayer, self).__init__(name=name)
        # mapping is a complete cache of the model file, the total number of
        # modalities are listed in self.modalities
        self.field = field
        self.modalities = modalities
        self.model_file = os.path.abspath(model_filename)
        self.label_map = hs.read_mapping_file(model_filename)

    @property
    def key(self):
        # provide a readable key for the label mapping item
        key_str = "{}_{}".format(self.field, self.modalities)
        return standardise_string(key_str)

    def layer_op(self, image, mask=None):
        assert self.is_ready(), \
            "discrete_label_normalisation layer needs to be trained first."
        # mask is not used for label mapping
        if isinstance(image, dict):
            label_data = np.asarray(image[self.field])
        else:
            label_data = np.asarray(image)

        mapping = self.label_map[self.key]
        assert len(np.unique(label_data)) <= len(mapping), \
            "couldn't find a unique mapping for discrete label maps, " \
            " please check the line starting with {} in {}, " \
            " remove the line to find the model again, or " \
            " check the input label image".format(self.key, self.model_file)
        mapped_label = np.copy(label_data)
        for (new_id, original) in enumerate(mapping):
            mapped_label[label_data == original] = new_id

        if isinstance(image, dict):
            image[self.field] = mapped_label
            return image, mask
        else:
            return mapped_label, mask

    def reverse_op(self, image, mask=None):
        assert self.is_ready(), \
            "discrete_label_normalisation layer needs to be trained first."
        # mask is not used for label mapping
        if isinstance(image, dict):
            label_data = np.asarray(image[self.field])
        else:
            label_data = np.asarray(image)

        mapping = self.label_map[self.key]
        assert len(np.unique(label_data)) <= len(mapping), \
            "couldn't find a unique mapping for discrete label maps, " \
            " please check the line starting with {} in {}".format(
                self.key, self.model_file)
        mapped_label = np.copy(label_data)
        for (new_id, original) in enumerate(mapping):
            mapped_label[label_data == new_id] = original

        if isinstance(image, dict):
            image[self.field] = mapped_label
            return image, mask
        else:
            return mapped_label, mask

    def is_ready(self):
        mapping = self.label_map.get(self.key, None)
        return True if mapping is not None else False

    def train(self, image_list):
        # check modalities to train, using the first subject in subject list
        # to find input modality list
        if self.is_ready():
            tf.logging.info("label mapping ready for {} n classes: {}".format(
                self.field, len(self.label_map[self.key])))
            return
        tf.logging.info(
            "Looking for the set of unique discrete labels from input {}"
            " using {} subjects".format(self.field, len(image_list)))
        label_map = find_set_of_labels(image_list, self.field, self.key)
        # merging trained_mapping dict and self.mapping dict
        self.label_map.update(label_map)
        hs.write_all_mod_mapping(self.model_file, self.label_map)


def find_set_of_labels(image_list, field, output_key):
    label_set = set()
    for idx, image in enumerate(image_list):
        printProgressBar(idx, len(image_list),
                         prefix='searching unique labels from training files',
                         decimals=1, length=10, fill='*')
        unique_label = np.unique(image[field].get_data(),
                                 return_index=False,
                                 return_inverse=False,
                                 return_counts=False,
                                 axis=None)
        label_set.update(set(unique_label))
    label_set = list(label_set)
    label_set.sort()
    return {output_key: tuple(label_set)}
