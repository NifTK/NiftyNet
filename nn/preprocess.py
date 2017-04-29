# -*- coding: utf-8 -*-
import os
import pickle
import sys

import numpy as np
from medpy.filter import IntensityRangeStandardization
from scipy.interpolate.interpolate import interp1d
import scipy.ndimage as nd


N_INTERVAL = 20

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
            return (img - np.mean(img)) / np.std(img)
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

        img = (img - np.mean(img)) / np.std(img)
        return img


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
            return np.tile(np.expand_dim(self.mask, 3), [1, 1, 1, img.shape[3]])
        img_masked = ma.masked_array(img,mask=self.mask)
        values_perc = scipy.stats.mstats.mquantiles(img_masked.flatten(), [self.perc_threshold, 1-self.perc_threshold])
        mask = np.copy(self.mask)
        mask[img_masked > np.max(values_perc)] = 1
        mask[img_masked < np.min(values_perc)] = 1
        print(np.count_nonzero(mask), np.count_nonzero(self.mask), values_perc)
        return mask
