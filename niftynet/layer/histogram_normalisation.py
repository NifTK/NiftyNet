# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import numpy as np

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
                 models_filename,
                 binary_masking_func=None,
                 norm_type='percentile',
                 cutoff=(0.05, 0.95),
                 name='hist_norm'):

        super(HistogramNormalisationLayer, self).__init__(name=name)
        self.hist_model_file = os.path.abspath(models_filename)

        if binary_masking_func is not None:
            assert isinstance(binary_masking_func, BinaryMaskingLayer)
            self.binary_masking_func = binary_masking_func
        self.norm_type = norm_type
        self.cutoff = cutoff

        # mapping is a complete cache of the model file, the total number of
        # modalities are listed in self.modalities
        self.mapping = hs.read_mapping_file(models_filename)
        self.modalities = {}

    def layer_op(self, image_5d, mask=None):
        assert image_5d.ndim == 5
        image_5d = np.asarray(image_5d, dtype=float)

        image_mask = None
        if mask is not None:
            image_mask = np.asarray(mask, dtype=np.bool)
        else:
            if self.binary_masking_func is not None:
                image_mask = self.binary_masking_func(image_5d)

        # no access to mask, default to all foreground
        if image_mask is None:
            image_mask = np.ones_like(image_5d, dtype=np.bool)

        normalised = self.__normalise_5d(image_5d, image_mask)
        return normalised, image_mask

    def __check_modalities_to_train(self, subjects):
        # collect all modality list from subjects
        for subject in subjects:
            self.modalities.update(subject.modalities_dict())
        if self.mapping is {}:
            return self.modalities
        # remove if exists in currently loaded mapping dict
        modalities_to_train = dict(self.modalities)
        for mod in self.modalities.keys():
            if mod in self.mapping:
                del modalities_to_train[mod]
        return modalities_to_train

    def is_ready(self, subjects):
        mod_to_train = self.__check_modalities_to_train(subjects)
        if len(mod_to_train) > 0:
            print('histogram normalisation, looking for reference histogram...')
            return False
        return True

    def train_normalisation_ref(self, subjects):
        # check modalities to train, using the first subject in subject list
        # to find input modality list
        mod_to_train = self.__check_modalities_to_train(subjects)
        if len(mod_to_train) == 0:
            print('Normalisation histogram reference models found')
            return
        array_files = [subject.column(0) for subject in subjects]
        print("training normalisation histogram references for {}, "
              "using {} subjects".format(mod_to_train.keys(), len(array_files)))
        trained_mapping = hs.create_mapping_from_multimod_arrayfiles(
            array_files, mod_to_train, self.cutoff, self.binary_masking_func)

        # merging trained_mapping dict and self.mapping dict
        self.mapping.update(trained_mapping)
        self.__write_all_mod_mapping()

    def __normalise_5d(self, data_array, mask_array):
        assert not self.modalities == {}
        assert data_array.ndim == 5
        assert data_array.shape[3] <= len(self.modalities)

        if self.mapping is {}:
            raise RuntimeError("calling normaliser with empty mapping,"
                               "probably {} is not loaded".format(
                self.hist_model_file))
        for mod in self.modalities:
            for t in range(0, data_array.shape[4]):
                mod_id = self.modalities[mod]
                if not np.any(data_array[..., mod_id, t]):
                    continue  # missing modality
                data_array[..., mod_id, t] = self.__normalise_3d(
                    data_array[..., mod_id, t],
                    mask_array[..., mod_id, t],
                    self.mapping[mod])
        return data_array

    def __normalise_3d(self, img_data, mask, mapping):
        assert img_data.ndim == 3
        assert np.all(img_data.shape[:3] == mask.shape[:3])

        return hs.transform_by_mapping(img_data,
                                       mask,
                                       mapping,
                                       self.cutoff,
                                       self.norm_type)

    # Function to modify the model file with the mapping if needed according
    # to existent mapping and modalities
    def __write_all_mod_mapping(self):
        # backup existing file first
        if os.path.exists(self.hist_model_file):
            backup_name = '{}.backup'.format(self.hist_model_file)
            from shutil import copyfile
            try:
                copyfile(self.hist_model_file, backup_name)
            except OSError:
                print('cannot backup file {}'.format(self.hist_model_file))
                raise
            print("moved existing histogram reference file\n"
                  " from {} to {}".format(self.hist_model_file, backup_name))

        if not os.path.exists(os.path.dirname(self.hist_model_file)):
            try:
                os.makedirs(os.path.dirname(self.hist_model_file))
            except OSError:
                print('cannot create {}'.format(self.hist_model_file))
                raise
        hs.force_writing_new_mapping(self.hist_model_file, self.mapping)
