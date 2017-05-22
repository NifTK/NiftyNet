# -*- coding: utf-8 -*-
import os
import pickle
import sys
import warnings
import numpy as np
import numpy.ma as ma
from medpy.filter import IntensityRangeStandardization
from scipy.interpolate.interpolate import interp1d
import scipy.ndimage as nd
import histogram_standardisation as hs
import utilities.misc as util
import utilities.misc_io as io
from shutil import copyfile


N_INTERVAL = 20


class HistNormaliser_bis(object):
    def __init__(self, models_filename, path_to_train, norm_type='percentile',
                 cutoff=[0.05, 0.95],
                 mask_type='otsu_plus', option_saving=''):
        self.models = models_filename
        self.path = [p for p in path_to_train]
        self.cutoff = cutoff
        self.norm_type = norm_type
        self.mask_type = mask_type
        self.option_saving = option_saving


    def training_normalisation_from_array_files(self, array_files,
                                                list_modalities):
        mapping = {}

        perc_database = hs.create_database_perc_multimod_arrayfiles(
            self.mask_type,array_files, self.cutoff, list_modalities)
        for m in perc_database.keys():
            s1, s2 = hs.create_standard_range(perc_database[m])
            print(perc_database, s1, s2)
            mapping[m] = hs.create_mapping_perc(perc_database[m], s1, s2)
        return mapping



    # To train the normalisation for the specified modalities
    def training_normalisation(self, modalities):
        mapping = {}
        for m in modalities:
            mapping[m] = []
            perc_database = hs.create_database_perc_dir(self.mask_type,
                                                        self.path, m,
                                                        self.cutoff,
                                                        self.norm_type)
            s1, s2 = hs.create_standard_range(perc_database)
            print(perc_database, s1, s2)
            mapping[m] = hs.create_mapping_perc(perc_database, s1, s2)
        return mapping

    # Function to modify the model file with the mapping if needed according
    # to existent mapping and modalities
    def complete_and_transform_model_file(self, mapping, modalities):
        modalities = [m for m in modalities if m in mapping.keys()]

        path, name, ext = io.split_filename(self.models)
        if not os.path.exists(path):
            os.mkdir(path)
        if os.path.exists(self.models):
            warnings.warn("The reference filename exists and will be copied "
                          "for modification")
            path, name, ext = io.split_filename(self.models)
            to_change = [m for m in modalities]
            to_change_string = map(str, modalities)
            new_name = os.path.join(path, name + '_' + ''.join(
                to_change_string)+'_'+self.option_saving+ext)


            with open(self.models) as oldfile, open(new_name, 'w+') as newfile:
                for line in oldfile:
                    if not any(m_to_change in line for m_to_change in
                               to_change_string):
                        newfile.write(line)
            self.models = new_name
        else:
            new_name = self.models
        for m in modalities:
            hs.write_mapping_file(mapping[m], new_name, m)
        return new_name

    def whitening_transformation(self, img, mask):
        # make sure img is a monomodal volume
        assert (len(img.shape) == 3) or (img.shape[3] == 1)
        masked_img = ma.masked_array(np.copy(img), 1-mask)
        mean = masked_img.mean()
        std = masked_img.std()
        img[mask == 1] -= mean
        img[mask == 1] /= std
        return img


    def normalise_data_array(self, data_array, mask_array):
        list_modalities = hs.list_trained_modalities(self.models)
        print("Modalities considered in the order %s" % ' '.join(
            list_modalities))
        data_array = io.expand_for_5d(data_array)
        if data_array.shape[3] > len(list_modalities):
            warnings.warn("There are more modalities to normalise than "
                             "reference histograms ! Please rerun the "
                             "histogram training")

            raise ValueError("There are more modalities to normalise than "
                             "reference histograms ! Please rerun the "
                             "histogram training")
        for mod in range(0, data_array.shape[3]):
            for t in range(0, data_array.shape[4]):
                if np.count_nonzero(data_array[..., mod, t]) == 0:
                    continue
                else:
                    if mask_array is None:
                        mask_temp = hs.create_mask_img_multimod(data_array[
                                                                    ...,mod,t],
                                                        self.mask_type)
                    else:
                        mask_temp = mask_array[...,mod,t]
                data_array[..., mod, t] = self.intensity_normalisation(
                    data_array[...,mod, t], mask_temp, list_modalities[mod])
        return data_array





    def intensity_normalisation(self, img_data, mask, modality):
        if not io.check_shape_compatibility_3d(img_data, mask):
            raise ValueError('incompatibility of shapes between img and mask')
        mapping = hs.read_mapping_file(self.models, modality)
        if len(mapping) == 15:
            final_mapping = mapping[1:-1]
        else:
            final_mapping = mapping
        if img_data.ndim == 3:
            mask_new = io.adapt_to_shape(mask, img_data.shape)
            new_img_temp = hs.transform_for_mapping(img_data,
                                                    mask_new,
                                                    final_mapping,
                                                    self.cutoff,
                                                    self.norm_type)
            return new_img_temp
        mask_new = io.adapt_to_shape(mask, img_data.shape)
        hs.standardise_cutoff(self.cutoff, self.norm_type)
        new_img = np.copy(img_data)
        for i in range(0, img_data.shape[-1]):

            # Histogram normalisation (foreground)
            new_img_temp = hs.transform_for_mapping(img_data[..., i],
                                                    mask_new[..., i],
                                                    final_mapping,
                                                    self.cutoff,
                                                    self.norm_type)
            # Whitening with zero mean and unit variance (foreground)
            # new_img_temp = self.whitening_transformation(new_img_temp,
            # mask_new[..., i])
            new_img[..., i] = new_img_temp
        return new_img

    def intensity_normalisation_multimod(self, img, modalities, mask):
        # first check that the length of modalities is the same as the number
        #  of modalities in 4th dimension of img
        if not img.shape[3] == len(modalities):
            raise ValueError('not same number of modalities as shape')
        if not io.check_shape_compatibility_3d(img, mask):
            raise ValueError('incompatibility of shapes between img and mask')
        hs.standardise_cutoff(self.cutoff, self.norm_type)
        new_img = np.copy(img)
        new_mask = io.adapt_to_shape(mask, img)
        for i in range(0, len(modalities)):
            # Histogram normalisation (foreground)
            new_img_temp = self.intensity_normalisation(
                new_img[:, :, :, i:i+1], new_mask[...,i:i+1], modalities[i])
            # Whitening with zero mean and unit variance (foreground)
            # new_img_temp = self.whitening_transformation(new_img_temp,
            # mask_new[..., i])
            new_img[..., i] = new_img_temp
        return new_img


