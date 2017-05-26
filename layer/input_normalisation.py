# -*- coding: utf-8 -*-
import os

import numpy as np
import numpy.ma as ma

import utilities.histogram_standardisation as hs
import utilities.misc_io as io
from .base import Layer


class HistogramNormalisationLayer(Layer):
    def __init__(self,
                 models_filename,
                 multimod_mask_type='or',
                 norm_type='percentile',
                 cutoff=[0.05, 0.95],
                 mask_type='otsu_plus',
                 name='hist_norm'):

        super(HistogramNormalisationLayer, self).__init__(name=name)
        self.hist_model_file = models_filename

        self.multimod_mask_type = multimod_mask_type
        self.norm_type = norm_type
        self.cutoff = cutoff
        self.mask_type = mask_type

        # mapping is a complete cache of the model file, the total number of
        # modalities are listed in self.modalities
        self.mapping = hs.read_mapping_file(models_filename)
        self.modalities = {}

    def layer_op(self, image_5d, do_normalising=False, do_whitening=False):
        if not (do_whitening and do_normalising):
            return image_5d

        mask_array = self.make_mask_array(image_5d)
        if do_normalising:
            image_5d = self.normalise(image_5d, mask_array)
        if do_whitening:
            image_5d = self.whiten(image_5d, mask_array)
        return image_5d

    def __set_modalities(self, subject):
        self.modalities = subject.modalities_dict()

    def __check_modalities_to_train(self):
        if self.mapping is {}:
            return self.modalities
        # remove if exists in currently loaded mapping dict
        modalities_to_train = dict(self.modalities)
        for mod in self.modalities.keys():
            if mod in self.mapping:
                del modalities_to_train[mod]
        return modalities_to_train

    def is_ready(self, do_normalisation, do_whitening):
        if not do_normalisation:
            return True # always ready for do_whitening
        if not (self.mapping and self.modalities):
            print 'histogram normalisation, looking for reference histogram...'
            return False
        return True

    def train_normalisation_ref(self, subjects):
        # check modalities to train, using the first subject in subject list
        # to find input modality list
        self.__set_modalities(subjects[0])
        mod_to_train = self.__check_modalities_to_train()
        if len(mod_to_train) <= 0:
            print('Normalisation histogram reference models found')
            return
        print('training normalisation histogram references for {}'.format(
            mod_to_train.keys()))
        array_files = [subject.column(0) for subject in subjects]
        trained_mapping = hs.create_mapping_from_multimod_arrayfiles(
            array_files, mod_to_train, self.cutoff, self.mask_type)
        ### merging trained_mapping dict and self.mapping dict
        # for python 3.5: self.mapping = {**self.mapping, **trained_mapping}
        self.mapping.update(trained_mapping)
        self.__write_all_mod_mapping()

    # TODO: handling mask output is all False
    def make_mask_array(self, data_array):
        # data_array = io.expand_to_5d(data_array)
        assert data_array.ndim == 5
        mod_to_mask = [m for m in range(0, data_array.shape[3]) if
                       np.any(data_array[..., m, :])]
        mask_array = np.zeros_like(data_array, dtype=bool)
        for mod in mod_to_mask:
            for t in range(0, data_array.shape[4]):
                mask_array[..., mod, t] = hs.create_mask_img_3d(
                    data_array[..., mod, t],
                    self.mask_type)

        if self.multimod_mask_type is None:
            return mask_array

        if self.multimod_mask_type == '':
            return mask_array

        if self.multimod_mask_type == 'all':
            return mask_array

        if self.multimod_mask_type == 'or':
            for t in range(0, data_array.shape[4]):
                new_mask = np.zeros(data_array.shape[0:3], dtype=np.bool)
                for mod in mod_to_mask:
                    new_mask = np.logical_or(new_mask, mask_array[..., mod, t])
                mask_array[..., t] = np.tile(np.expand_dims(new_mask, axis=-1),
                                             [1, mask_array.shape[3]])
            return mask_array

        if self.multimod_mask_type == 'and':
            for t in range(0, data_array.shape[4]):
                new_mask = np.ones(data_array.shape[0:3])
                for mod in mod_to_mask:
                    new_mask = np.logical_and(new_mask, mask_array[..., mod, t])
                mask_array[..., t] = np.tile(np.expand_dims(new_mask, axis=-1),
                                             [1, mask_array.shape[3]])
            return mask_array
        raise ValueError('unknown mask combining option')

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
        img[mask == True] /= std
        return img

    def normalise(self, data_array, mask_array):
        assert not self.modalities == {}
        assert data_array.ndim == 5
        assert data_array.shape[3] <= len(self.modalities)

        if self.mapping is {}:
            raise RuntimeError("calling normalisor with empty mapping,"
                "probably {} doesnot exists or not loaded".format(
                    self.hist_model_file))
        for mod in self.modalities.keys():
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

        #mask_new = io.adapt_to_shape(mask, img_data.shape)
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
            print("moved existing histogram refernce file\n"
                  " from {} to {}".format(self.hist_model_file, backup_name))

        if not os.path.exists(os.path.dirname(self.hist_model_file)):
            try:
                os.mkdirs(os.path.dirname(self.hist_model_file))
            except OSError:
                print('cannot create {}'.format(self.hist_model_file))
                raise
        hs.force_writing_new_mapping(self.hist_model_file, self.mapping)


# class HistNormaliser(object):
#    def __init__(self, ref_file_name):
#        self.ref_file_name = ref_file_name
#        self.irs_model = []
#        self.__init_precomputed_model()
#
#    def __init_precomputed_model(self):
#        self.irs_model = IntensityRangeStandardization()
#        if not os.path.exists(self.ref_file_name):
#            return
#        with open(self.ref_file_name, 'rb') as hist_ref:
#            if sys.version_info > (3, 0):
#                self.irs_model = pickle.load(hist_ref, encoding='latin1')
#            else:
#                self.irs_model = pickle.load(hist_ref)
#            print("Reference histogram loaded")
#
#    def intensity_normalisation(self, img, randomised=False):
#        if not os.path.exists(self.ref_file_name):
#            print("No histogram normalization")
#            fg = img > 0.0  # foreground
#            img_norm = img
#            img_norm[fg] = (img[fg] - np.mean(img[fg])) / np.std(img[fg])
#            return img_norm
#        bin_id = np.random.randint(0, N_INTERVAL) if randomised else -1
#
#        intensity_hist = np.histogram(img, 1000)
#        # edge of first mode in the histogram
#        first_mode = intensity_hist[1][np.argmax(intensity_hist[0]) + 1]
#        # divide values in between first mode and image_mean into N_INTERVAL
#        all_inter = np.linspace(first_mode, np.mean(img), N_INTERVAL)
#        # a 'foreground' mask by a threshold in [first_mode, image_mean]
#        mask = nd.morphology.binary_fill_holes(img >= all_inter[bin_id])
#
#        # compute landmarks from image foreground (by applying the mask)
#        li = np.percentile(img[mask == True],
#                              [self.irs_model.cutoffp[0]] +\
#                              self.irs_model.landmarkp +\
#                              [self.irs_model.cutoffp[1]])
#        # mapping from landmarks to the reference histogram
#        ipf = interp1d(li, self.irs_model.model, bounds_error=False)
#        # transform image
#        mapped_img = ipf(img)
#
#        # linear model on both open ends of the mapping
#        left_linearmodel = IntensityRangeStandardization.linear_model(
#            li[:2], self.irs_model.model[:2])
#        right_linearmodel = IntensityRangeStandardization.linear_model(
#            li[-2:], self.irs_model.model[-2:])
#        left_selector = img < li[0]
#        right_selector = img > li[-1]
#        img[left_selector] = left_linearmodel(img[left_selector])
#        img[right_selector] = right_linearmodel(img[right_selector])
#
#        fg = img > 0.0  # foreground
#        img_norm = img
#        img_norm[fg] = (img[fg] - np.mean(img[fg])) / np.std(img[fg])
#        return img_norm


# class MahalNormaliser(object):
#    def __init__(self, mask, perc_threshold):
#        self.mask = mask
#        self.perc_threshold = perc_threshold
#
#    def intensity_normalisation(self, img, normalisation_indices):
#        if img.ndim == 3:
#            img = np.expand_dims(img, 3)
#        for n in range(0, img.shape[3]):
#            img_temp = np.squeeze(img[:, :, :, n])
#            if n in normalisation_indices:
#                if self.perc_threshold == 0:
#                    mask_fin = self.mask
#                else:
#                    mask_fin = self.create_fin_mask(img_temp)
#                img_masked = ma.masked_array(img_temp, mask=mask_fin)
#                img_masked_mean = img_masked.mean()
#                img_masked_var = img_masked.var()
#                img[:, :, :, n] = np.expand_dims(np.sign(img_temp-img_masked_mean) *\
#                                  np.sqrt(np.square(img_temp-img_masked_mean)/img_masked_var), 3)
#            else:
#                img[:, :, :, n] = np.expand_dims(img_temp, 3)
#        return img
#
#    def create_fin_mask(self, img):
#        if img.ndim == 4:
#            return np.tile(np.expand_dims(self.mask, 3), [1, 1, 1, img.shape[
#                3]])
#        img_masked = ma.masked_array(img, mask=self.mask)
#        values_perc = scipy.stats.mstats.mquantiles(img_masked.flatten(),
#                                                    [self.perc_threshold, 1-self.perc_threshold])
#        mask = np.copy(self.mask)
#        mask[img_masked > np.max(values_perc)] = 1
#        mask[img_masked < np.min(values_perc)] = 1
#        print(np.count_nonzero(mask), np.count_nonzero(self.mask), values_perc)
#        return mask
