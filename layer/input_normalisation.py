# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import numpy as np
import numpy.ma as ma

import utilities.histogram_standardisation as hs
from .base_layer import Layer
from .binary_masking import BinaryMaskingLayer


class HistogramNormalisationLayer(Layer):
    def __init__(self,
                 models_filename,
                 binary_masking_func=None,
                 norm_type='percentile',
                 cutoff=(0.05, 0.95),
                 name='hist_norm'):

        super(HistogramNormalisationLayer, self).__init__(name=name)
        self.hist_model_file = models_filename

        if binary_masking_func is not None:
            assert isinstance(binary_masking_func, BinaryMaskingLayer)
            self.binary_masking_func = binary_masking_func
        self.norm_type = norm_type
        self.cutoff = cutoff

        # mapping is a complete cache of the model file, the total number of
        # modalities are listed in self.modalities
        self.mapping = hs.read_mapping_file(models_filename)
        self.modalities = {}

    def layer_op(self, image_5d, do_normalising=False, do_whitening=False):
        if not (do_whitening and do_normalising):
            return image_5d

        if self.binary_masking_func is not None:
            mask_array = self.binary_masking_func(image_5d)
        else:
            mask_array = np.ones_like(image_5d, dtype=np.bool)

        if do_normalising:
            image_5d = self.normalise(image_5d, mask_array)
        if do_whitening:
            image_5d = self.whiten(image_5d, mask_array)
        return image_5d

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

    def is_ready(self, subjects, do_normalisation, do_whitening):
        if not do_normalisation:
            return True  # always ready for do_whitening
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


    def whiten(self, data_array, mask_array):
        for m in range(0, data_array.shape[3]):
            for t in range(0, data_array.shape[4]):
                data_array[..., m, t] = \
                    self.whitening_transformation_3d(data_array[..., m, t],
                                                     mask_array[..., m, t])
        return data_array

    def whitening_transformation_3d(self, img, mask):
        # make sure img is a monomodal volume
        assert img.ndim == 3

        masked_img = ma.masked_array(np.copy(img), np.logical_not(mask))
        mean = masked_img.mean()
        std = masked_img.std()
        img[mask == True] -= mean
        img[mask == True] /= max(std, 1e-5)
        return img

    def normalise(self, data_array, mask_array):
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
                data_array[..., mod_id, t] = self.intensity_normalisation_3d(
                    data_array[..., mod_id, t],
                    mask_array[..., mod_id, t],
                    self.mapping[mod])
        return data_array

    def intensity_normalisation_3d(self, img_data, mask, mapping):
        assert img_data.ndim == 3
        assert np.all(img_data.shape[:3] == mask.shape[:3])

        # mask_new = io.adapt_to_shape(mask, img_data.shape)
        img_data = hs.transform_by_mapping(img_data,
                                           mask,
                                           mapping,
                                           self.cutoff,
                                           self.norm_type)
        return img_data

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
