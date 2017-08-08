# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import os

import numpy as np
import tensorflow as tf

import niftynet.utilities.histogram_standardisation as hs
from niftynet.layer.base_layer import DataDependentLayer
from niftynet.layer.binary_masking import BinaryMaskingLayer

"""
This class computes histogram based normalisation. A `training`
process is first used to find an averaged histogram mapping
from all training volumes.  This layer maintains the mapping array,
and the layer_op maps the intensity of new volumes to a normalised version.
The histogram is computed from foreground if a definition is provided for
foreground (by `binary_masking_func` or a `mask` matrix)
"""


class HistogramNormalisationLayer(DataDependentLayer):
    def __init__(self,
                 field,
                 modalities,
                 model_filename,
                 binary_masking_func=None,
                 norm_type='percentile',
                 cutoff=(0.05, 0.95),
                 name='hist_norm'):
        """

        :param field:
        :param modalities:
        :param model_filename:
        :param binary_masking_func: set to None for global mapping
        :param norm_type:
        :param cutoff:
        :param name:
        """

        super(HistogramNormalisationLayer, self).__init__(name=name)
        self.hist_model_file = os.path.abspath(model_filename)

        if binary_masking_func is not None:
            assert isinstance(binary_masking_func, BinaryMaskingLayer)
            self.binary_masking_func = binary_masking_func
        self.norm_type = norm_type
        self.cutoff = cutoff
        self.field = field

        # mapping is a complete cache of the model file, the total number of
        # modalities are listed in self.modalities tuple
        self.mapping = hs.read_mapping_file(model_filename)
        self.modalities = modalities

    def layer_op(self, image, mask=None):
        assert self.is_ready(), \
            "histogram normalisation layer needs to be trained first."
        if isinstance(image, dict):
            image_5d = np.asarray(image[self.field], dtype=np.float32)
        else:
            image_5d = np.asarray(image, dtype=np.float32)

        image_mask = None
        if isinstance(mask, dict):
            image_mask = mask.get(self.field, None)
        elif mask is not None:
            image_mask = mask
        elif self.binary_masking_func is not None:
            image_mask = self.binary_masking_func(image_5d)
        else:
            # no access to mask, default to all image
            image_mask = np.ones_like(image_5d, dtype=np.bool)

        normalised = self._normalise_5d(image_5d, image_mask)

        if isinstance(image, dict):
            image[self.field] = normalised
            if isinstance(mask, dict):
                mask[self.field] = image_mask
            else:
                mask = {self.field: image_mask}
            return image, mask
        else:
            return normalised, image_mask

    def __check_modalities_to_train(self):
        modalities_to_train = [mod for mod in self.modalities
                               if not mod in self.mapping]
        return set(modalities_to_train)

    def is_ready(self):
        mod_to_train = self.__check_modalities_to_train()
        return False if mod_to_train else True

    def train(self, image_list):
        # check modalities to train, using the first subject in subject list
        # to find input modality list
        if self.is_ready():
            tf.logging.info("Normalisation histogram reference models ready"
                            " for {}:{}".format(self.field, self.modalities))
            return
        mod_to_train = self.__check_modalities_to_train()
        tf.logging.info(
            "training normalisation histogram references "
            "for {}:{}, using {} subjects".format(
                self.field, mod_to_train, len(image_list)))
        trained_mapping = hs.create_mapping_from_multimod_arrayfiles(
            image_list, self.field, self.modalities, mod_to_train,
            self.cutoff, self.binary_masking_func)

        # merging trained_mapping dict and self.mapping dict
        self.mapping.update(trained_mapping)
        hs.write_all_mod_mapping(self.hist_model_file, self.mapping)

    def _normalise_5d(self, data_array, mask_array):
        assert self.modalities
        assert data_array.ndim == 5
        assert data_array.shape[4] <= len(self.modalities)

        if not self.mapping:
            tf.logging.fatal(
                "calling normaliser with empty mapping,"
                "probably {} is not loaded".format(self.hist_model_file))
            raise RuntimeError
        mask_array = np.asarray(mask_array, dtype=np.bool)
        for mod_id, mod_name in enumerate(self.modalities):
            if not np.any(data_array[..., mod_id]):
                continue  # missing modality
            data_array[..., mod_id] = self.__normalise(
                data_array[..., mod_id],
                mask_array[..., mod_id],
                self.mapping[mod_name])
        return data_array

    def __normalise(self, img_data, mask, mapping):
        return hs.transform_by_mapping(
            img_data, mask, mapping, self.cutoff, self.norm_type)
