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
import utilities.misc_io as io
from shutil import copyfile


N_INTERVAL = 20


class HistNormaliser_bis(object):
    def __init__(self, models_filename, path_to_train,
                 dict_masking, norm_type='percentile',
                 cutoff=[0.05, 0.95],
                 mask_type='otsu_plus', option_saving=''):
        self.models = models_filename
        self.path = [p for p in path_to_train]
        self.cutoff = cutoff
        self.norm_type = norm_type
        self.mask_type = mask_type
        self.option_saving = option_saving
        self.dict_masking = dict_masking

    # Retraining of the standardisation if needed
    def retrain_standardisation(self, flag_retrain, modalities):
        if flag_retrain:
            mapping = self.training_normalisation(modalities)
            new_model = self.complete_and_transform_model_file(mapping,
                                                               modalities)
            self.models = new_model
        else:
            modalities_to_train = hs.check_modalities_to_train(self.models,
                                                               modalities)
            if len(modalities_to_train) > 0:
                modalities = modalities_to_train
                mapping = self.training_normalisation(modalities)
                if mapping is not None:
                    new_model = self.complete_and_transform_model_file(
                                    mapping, modalities_to_train)
                    self.models = new_model

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

    def make_mask_array(self, data_array, flag_all=False):
        data_array = io.expand_for_5d(data_array)
        max_time = data_array.shape[4]
        list_indices_mod = [m for m in range(0, data_array.shape[3]) if
                            np.count_nonzero(data_array[..., m, :]) > 0]
        mod_masking = list_indices_mod
        mask_array = np.zeros_like(data_array)
        for mod in mod_masking:
            for t in range(0, max_time):
                new_mask = hs.create_mask_img_3d(
                    data_array[..., mod, t], self.dict_masking.mask_type)
                new_mask = io.expand_for_5d(new_mask)
                mask_array[..., mod:mod+1, t:t+1] = new_mask
        if flag_all:
            return mask_array
        elif self.dict_masking.multimod_type == 'or':
            for t in range(0, max_time):
                new_mask = np.zeros([data_array.shape[0:3]])
                for mod in mod_masking:
                    if np.count_nonzero(data_array[..., mod, t]) > 0:
                        new_mask = new_mask + mask_array[..., mod, t]
                new_mask[new_mask > 0.5] = 1
                mask_array[..., t] = io.expand_for_5d(np.tile(np.expand_dims(
                    new_mask, axis=3), [1, mask_array.shape[3]]))
            return mask_array
        elif self.dict_masking.multimod_type == 'and':
            for t in range(0, max_time):
                new_mask = np.ones(data_array.shape[0:3])
                for mod in mod_masking:
                    if np.count_nonzero(data_array[..., mod, t]) > 0:
                        new_mask = np.multiply(new_mask,
                                               mask_array[..., mod, t])
                mask_array[..., t:t+1] = io.expand_for_5d(np.tile(
                    np.expand_dims(new_mask, axis=3), [1, mask_array.shape[3]]))
            return mask_array
        else:
            return mask_array

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
        if mask_array is None:
            mask_array = self.make_mask_array(data_array, flag_all='True')
        for mod in range(0, data_array.shape[3]):
            for t in range(0, data_array.shape[4]):
                if np.count_nonzero(data_array[..., mod, t]) == 0:
                    continue
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


class HistNormaliser(object):
    def __init__(self, ref_file_name):
        self.ref_file_name = ref_file_name
        self.irs_model = []
        self.__init_precomputed_model()

    def __init_precomputed_model(self):
        self.irs_model = IntensityRangeStandardization()
        if not os.path.exists(self.ref_file_name):
            return
        with open(self.ref_file_name, 'rb') as hist_ref:
            if sys.version_info > (3, 0):
                self.irs_model = pickle.load(hist_ref, encoding='latin1')
            else:
                self.irs_model = pickle.load(hist_ref)
            print("Reference histogram loaded")

    def intensity_normalisation(self, img, randomised=False):
        if not os.path.exists(self.ref_file_name):
            print("No histogram normalization")
            fg = img > 0.0  # foreground
            img_norm = img
            img_norm[fg] = (img[fg] - np.mean(img[fg])) / np.std(img[fg])
            return img_norm
        bin_id = np.random.randint(0, N_INTERVAL) if randomised else -1

        intensity_hist = np.histogram(img, 1000)
        # edge of first mode in the histogram
        first_mode = intensity_hist[1][np.argmax(intensity_hist[0]) + 1]
        # divide values in between first mode and image_mean into N_INTERVAL
        all_inter = np.linspace(first_mode, np.mean(img), N_INTERVAL)
        # a 'foreground' mask by a threshold in [first_mode, image_mean]
        mask = nd.morphology.binary_fill_holes(img >= all_inter[bin_id])

        # compute landmarks from image foreground (by applying the mask)
        li = np.percentile(img[mask == True],
                              [self.irs_model.cutoffp[0]] +\
                              self.irs_model.landmarkp +\
                              [self.irs_model.cutoffp[1]])
        # mapping from landmarks to the reference histogram
        ipf = interp1d(li, self.irs_model.model, bounds_error=False)
        # transform image
        mapped_img = ipf(img)

        # linear model on both open ends of the mapping
        left_linearmodel = IntensityRangeStandardization.linear_model(
            li[:2], self.irs_model.model[:2])
        right_linearmodel = IntensityRangeStandardization.linear_model(
            li[-2:], self.irs_model.model[-2:])
        left_selector = img < li[0]
        right_selector = img > li[-1]
        img[left_selector] = left_linearmodel(img[left_selector])
        img[right_selector] = right_linearmodel(img[right_selector])

        fg = img > 0.0  # foreground
        img_norm = img
        img_norm[fg] = (img[fg] - np.mean(img[fg])) / np.std(img[fg])
        return img_norm


class MahalNormaliser(object):
    def __init__(self, mask, perc_threshold):
        self.mask = mask
        self.perc_threshold = perc_threshold

    def intensity_normalisation(self, img, normalisation_indices):
        if img.ndim == 3:
            img = np.expand_dims(img, 3)
        for n in range(0, img.shape[3]):
            img_temp = np.squeeze(img[:, :, :, n])
            if n in normalisation_indices:
                if self.perc_threshold == 0:
                    mask_fin = self.mask
                else:
                    mask_fin = self.create_fin_mask(img_temp)
                img_masked = ma.masked_array(img_temp, mask=mask_fin)
                img_masked_mean = img_masked.mean()
                img_masked_var = img_masked.var()
                img[:, :, :, n] = np.expand_dims(np.sign(img_temp-img_masked_mean) *\
                                  np.sqrt(np.square(img_temp-img_masked_mean)/img_masked_var), 3)
            else:
                img[:, :, :, n] = np.expand_dims(img_temp, 3)
        return img

    def create_fin_mask(self, img):
        if img.ndim == 4:
            return np.tile(np.expand_dims(self.mask, 3), [1, 1, 1, img.shape[
                3]])
        img_masked = ma.masked_array(img, mask=self.mask)
        values_perc = scipy.stats.mstats.mquantiles(img_masked.flatten(),
                                                    [self.perc_threshold, 1-self.perc_threshold])
        mask = np.copy(self.mask)
        mask[img_masked > np.max(values_perc)] = 1
        mask[img_masked < np.min(values_perc)] = 1
        print(np.count_nonzero(mask), np.count_nonzero(self.mask), values_perc)
        return mask
