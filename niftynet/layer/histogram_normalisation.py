# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import numpy as np
import tensorflow as tf

import niftynet.utilities.histogram_standardisation as hs
from niftynet.layer.base_layer import Layer
from niftynet.layer.binary_masking import BinaryMaskingLayer

"""
This class computes histogram based normalisation. A `training`
process is first used to find an averaged histogram mapping
from all training volumes.  This layer maintains the mapping array,
and the layer_op maps the intensity of new volumes to a normalised version.
The histogram is computed from foreground if a definition is provided for
foreground (by `binary_masking_func` or a `mask` matrix)
"""


class HistogramNormalisationLayer(Layer):
    def __init__(self,
                 field,
                 modalities,
                 models_filename,
                 binary_masking_func=None,
                 norm_type='percentile',
                 cutoff=(0.05, 0.95),
                 name='hist_norm'):
        """

        :param field:
        :param modalities:
        :param models_filename:
        :param binary_masking_func: set to None for global mapping
        :param norm_type:
        :param cutoff:
        :param name:
        """

        super(HistogramNormalisationLayer, self).__init__(name=name)
        self.hist_model_file = os.path.abspath(models_filename)

        if binary_masking_func is not None:
            assert isinstance(binary_masking_func, BinaryMaskingLayer)
            self.binary_masking_func = binary_masking_func
        self.norm_type = norm_type
        self.cutoff = cutoff
        self.field = field

        # mapping is a complete cache of the model file, the total number of
        # modalities are listed in self.modalities tuple
        self.mapping = hs.read_mapping_file(models_filename)
        self.modalities = modalities

    def layer_op(self, image_5d, mask=None):
        assert image_5d.ndim == 5
        image_5d = np.asarray(image_5d, dtype=float)

        image_mask = None
        if mask is not None:
            image_mask = np.asarray(mask, dtype=np.bool)
        else:
            if self.binary_masking_func is not None:
                image_mask = self.binary_masking_func(image_5d)

        # no access to mask, default to all image
        if image_mask is None:
            image_mask = np.ones_like(image_5d, dtype=np.bool)

        normalised = self.__normalise_5d(image_5d, image_mask)
        return normalised, image_mask

    def __check_modalities_to_train(self):
        modalities_to_train = [mod for mod in self.modalities
                               if not mod in self.mapping]
        return set(modalities_to_train)

    def is_ready(self):
        mod_to_train = self.__check_modalities_to_train()
        if mod_to_train:
            tf.logging.info('histogram normalisation, '
                            'looking for reference histogram...')
            return False
        return True

    def train(self, image_list):
        # check modalities to train, using the first subject in subject list
        # to find input modality list
        mod_to_train = self.__check_modalities_to_train()
        if len(mod_to_train) == 0:
            tf.logging.info('Normalisation histogram reference models found')
            return
        tf.logging.info("training normalisation histogram references for {}, "
                        "using {} subjects".format(mod_to_train,
                                                   len(image_list)))
        trained_mapping = hs.create_mapping_from_multimod_arrayfiles(
            image_list, self.field, self.modalities, mod_to_train,
            self.cutoff, self.binary_masking_func)

        # merging trained_mapping dict and self.mapping dict
        self.mapping.update(trained_mapping)
        hs.write_all_mod_mapping(self.hist_model_file, self.mapping)

    def __normalise_5d(self, data_array, mask_array):
        assert not self.modalities == {}
        assert data_array.ndim == 5
        assert data_array.shape[3] <= len(self.modalities)

        if not self.mapping:
            tf.logging.fatal(
                "calling normaliser with empty mapping,"
                "probably {} is not loaded".format(self.hist_model_file))
            raise RuntimeError
        for t in range(0, data_array.shape[3]):
            for mod_id, mod_name in enumerate(self.modalities):
                if not np.any(data_array[..., t, mod_id]):
                    continue  # missing modality
                data_array[..., t, mod_id] = self.__normalise_3d(
                    data_array[..., t, mod_id],
                    mask_array[..., t, mod_id],
                    self.mapping[mod_name])
        return data_array

    def __normalise_3d(self, img_data, mask, mapping):
        assert img_data.ndim == 3
        assert np.all(img_data.shape[:3] == mask.shape[:3])

        return hs.transform_by_mapping(img_data,
                                       mask,
                                       mapping,
                                       self.cutoff,
                                       self.norm_type)
