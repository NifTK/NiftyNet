# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import os

import numpy as np
import tensorflow as tf

import niftynet.utilities.histogram_standardisation as hs
from niftynet.layer.base_layer import DataDependentLayer
from niftynet.layer.base_layer import Invertible
from niftynet.utilities.user_parameters_helper import standardise_string
from niftynet.utilities.util_common import print_progress_bar


class DiscreteLabelNormalisationLayer(DataDependentLayer, Invertible):
    def __init__(self,
                 image_name,
                 modalities,
                 model_filename,
                 name='label_norm'):

        super(DiscreteLabelNormalisationLayer, self).__init__(name=name)
        # mapping is a complete cache of the model file, the total number of
        # modalities are listed in self.modalities
        self.image_name = image_name
        self.modalities = modalities
        self.model_file = os.path.abspath(model_filename)
        assert not os.path.isdir(self.model_file), \
            "model_filename is a directory, please change histogram_ref_file"
        self.label_map = hs.read_mapping_file(model_filename)

    @property
    def key(self):
        # provide a readable key for the label mapping item
        key_str = "{}_{}".format(self.image_name, self.modalities)
        return standardise_string(key_str)

    def layer_op(self, image, mask=None):
        assert self.is_ready(), \
            "discrete_label_normalisation layer needs to be trained first."
        # mask is not used for label mapping
        if isinstance(image, dict):
            if self.image_name not in image:
                return image, mask
            label_data = np.asarray(image[self.image_name])
        else:
            label_data = np.asarray(image)

        mapping = self.label_map[self.key]
        assert len(np.unique(label_data)) <= len(mapping), \
            "couldn't find a unique mapping for discrete label maps, " \
            " please check the line starting with {} in {}, " \
            " remove the line to find the model again, or " \
            " check the input label image".format(self.key, self.model_file)
        # map_dict = {}
        # for new_id, original in enumerate(mapping):
        #    map_dict[original] = new_id
        # mapped_data = np.vectorize(map_dict.get)(label_data)
        image_shape = label_data.shape
        label_data = label_data.reshape(-1)
        mapped_data = np.zeros_like(label_data)
        for (new_id, original) in enumerate(mapping):
            mapped_data[label_data == original] = new_id
        label_data = mapped_data.reshape(image_shape)

        if isinstance(image, dict):
            image[self.image_name] = label_data
            return image, mask
        return label_data, mask

    def inverse_op(self, image, mask=None):
        assert self.is_ready(), \
            "discrete_label_normalisation layer needs to be trained first."
        # mask is not used for label mapping
        if isinstance(image, dict):
            label_data = np.asarray(image[self.image_name])
        else:
            label_data = np.asarray(image)

        mapping = self.label_map[self.key]
        assert len(np.unique(label_data)) <= len(mapping), \
            "couldn't find a unique mapping for discrete label maps, " \
            " please check the line starting with {} in {}".format(
                self.key, self.model_file)
        image_shape = label_data.shape
        label_data = label_data.reshape(-1)
        mapped_data = np.zeros_like(label_data)
        for (new_id, original) in enumerate(mapping):
            mapped_data[label_data == new_id] = original
        label_data = mapped_data.reshape(image_shape)
        if isinstance(image, dict):
            image[self.image_name] = label_data
            return image, mask
        return label_data, mask

    def is_ready(self):
        mapping = self.label_map.get(self.key, None)
        return True if mapping is not None else False

    def train(self, image_list):
        # check modalities to train, using the first subject in subject list
        # to find input modality list
        assert image_list is not None, "nothing to training for this layer"
        if self.is_ready():
            tf.logging.info(
                "label mapping ready for {}:{}, {} classes".format(
                    self.image_name,
                    self.modalities,
                    len(self.label_map[self.key])))
            return
        tf.logging.info(
            "Looking for the set of unique discrete labels from input {}"
            " using {} subjects".format(self.image_name, len(image_list)))
        label_map = find_set_of_labels(image_list, self.image_name, self.key)
        # merging trained_mapping dict and self.mapping dict
        self.label_map.update(label_map)
        all_maps = hs.read_mapping_file(self.model_file)
        all_maps.update(self.label_map)
        hs.write_all_mod_mapping(self.model_file, all_maps)


def find_set_of_labels(image_list, field, output_key):
    label_set = set()
    for idx, image in enumerate(image_list):
        assert field in image, \
            "no {} data provided in for label mapping".format(field)
        print_progress_bar(idx, len(image_list),
                           prefix='searching unique labels from training files',
                           decimals=1, length=10, fill='*')
        unique_label = np.unique(image[field].get_data())
        if len(unique_label) > 500 or len(unique_label) <= 1:
            tf.logging.warning(
                'unusual values: number unique '
                'labels to normalise %s', len(unique_label))
        label_set.update(set(unique_label))
    label_set = list(label_set)
    label_set.sort()
    return {output_key: tuple(label_set)}
