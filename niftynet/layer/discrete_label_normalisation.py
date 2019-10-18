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
                 model_filename=None,
                 name='label_norm'):

        super(DiscreteLabelNormalisationLayer, self).__init__(name=name)
        # mapping is a complete cache of the model file, the total number of
        # modalities are listed in self.modalities
        self.image_name = image_name
        self.modalities = None
        if isinstance(modalities, (list, tuple)):
            if len(modalities) > 1:
                raise NotImplementedError(
                    "Currently supports single modality discrete labels.")
            self.modalities = modalities
        else:
            self.modalities = (modalities,)
        if model_filename is None:
            model_filename = os.path.join('.', 'histogram_ref_file.txt')
        self.model_file = os.path.abspath(model_filename)
        self._key=None
        assert not os.path.isdir(self.model_file), \
            "model_filename is a directory, " \
            "please change histogram_ref_file to a filename."
        self.label_map = hs.read_mapping_file(self.model_file)

    @property
    def key(self):
        if self._key:
            return self._key
        # provide a readable key for the label mapping item
        name1 = self.image_name
        name2 = self.image_name if not self.modalities else self.modalities[0]
        key_from = "{}_{}-from".format(name1, name2)
        key_to = "{}_{}-to".format(name1, name2)
        return standardise_string(key_from), standardise_string(key_to)

    @key.setter
    def key(self, value):
        # Allows the key to be overridden
        self._key=value

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

        mapping_from = self.label_map[self.key[0]]
        mapping_to = self.label_map[self.key[1]]

        image_shape = label_data.shape
        label_data = label_data.reshape(-1)
        mapped_data = np.zeros_like(label_data)
        for (original, new_id) in zip(mapping_from, mapping_to):
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

        mapping_from = self.label_map[self.key[0]]
        mapping_to = self.label_map[self.key[1]]

        image_shape = label_data.shape
        label_data = label_data.reshape(-1)
        mapped_data = np.zeros_like(label_data)
        for (new_id, original) in zip(mapping_from, mapping_to):
            mapped_data[label_data == original] = new_id
        label_data = mapped_data.reshape(image_shape)
        if isinstance(image, dict):
            image[self.image_name] = label_data
            return image, mask
        return label_data, mask

    def is_ready(self):
        mapping_from = self.label_map.get(self.key[0], None)
        if mapping_from is None:
            # tf.logging.warning('could not find mapping key %s', self.key[0])
            return False
        mapping_to = self.label_map.get(self.key[1], None)
        if mapping_to is None:
            # tf.logging.warning('could not find mapping key %s', self.key[1])
            return False
        assert len(mapping_from) == len(mapping_to), \
            "mapping is not one-to-one, " \
            "corrupted mapping file? {}".format(self.model_file)
        return True

    def train(self, image_list):
        # check modalities to train, using the first subject in subject list
        # to find input modality list
        assert image_list is not None, "nothing to training for this layer"
        if self.is_ready():
            tf.logging.info(
                "label mapping ready for {}:{}, {} classes".format(
                    self.image_name,
                    self.modalities,
                    len(self.label_map[self.key[0]])))
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
    if field in image_list[0] :
        for idx, image in enumerate(image_list):
            assert field in image, \
                "label normalisation layer requires {} input, " \
                "however it is not provided in the config file.\n" \
                "Please consider setting " \
                "label_normalisation to False.".format(field)
            print_progress_bar(idx, len(image_list),
                               prefix='searching unique labels from files',
                               decimals=1, length=10, fill='*')
            unique_label = np.unique(image[field].get_data())
            if len(unique_label) > 500 or len(unique_label) <= 1:
                tf.logging.warning(
                    'unusual discrete values: number of unique '
                    'labels to normalise %s', len(unique_label))
            label_set.update(set(unique_label))
        label_set = list(label_set)
        label_set.sort()
    try:
        mapping_from_to = dict()
        mapping_from_to[output_key[0]] = tuple(label_set)
        mapping_from_to[output_key[1]] = tuple(range(0, len(label_set)))
    except (IndexError, ValueError):
        tf.logging.fatal("unable to create mappings keys: %s, image name %s",
                         output_key, field)
        raise
    return mapping_from_to
