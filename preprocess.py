# -*- coding: utf-8 -*-
import os
import pickle
import sys
import scipy.stats.mstats

import numpy as np
import numpy.ma as ma

from external.intensity_range_standardization import IntensityRangeStandardization


class HistNormaliser(object):
    def __init__(self, ref_file_name):
        self.ref_file_name = ref_file_name
        self.irs_model = []
        self.__init_precomputed_model()

    def __init_precomputed_model(self):
        self.irs_model = IntensityRangeStandardization()
        if self.ref_file_name is "":
            return
        with open(self.ref_file_name, 'rb') as hist_ref:
            if sys.version_info > (3, 0):
                self.irs_model = pickle.load(hist_ref, encoding='latin1')
            else:
                self.irs_model = pickle.load(hist_ref)
            print("Reference histogram loaded")

    def intensity_normalisation(self, img, randomised=False):
        if self.ref_file_name is "":
            return (img - np.mean(img)) / np.std(img)
        bin_id = np.random.randint(0, 20) if randomised else -1
        img = self.irs_model.transform(img, thr=bin_id)
        img = (img - np.mean(img)) / np.std(img)
        return img

class MahalNormaliser(object):
    def __init__(self, mask, perc_threshold):
        self.mask = mask
        self.perc_threshold = perc_threshold

    def mahalanobis_normalisation(self,img):
        if img.ndim == 3:
            img = np.expand_dims(img,3)
        for n in range(0,img.shape[3]):
            img_temp = np.squeeze(img[:,:,:,n])
            if self.perc_threshold == 0:
                mask_fin = self.mask
            else:
                mask_fin = self.create_fin_mask(img_temp)
            img_masked = ma.masked_array(img_temp, mask=mask_fin)
            img_masked_mean = img_masked.mean()
            img_masked_var = img_masked.var()
            img[:,:,:,n] = np.sign(img_temp-img_masked_mean)*np.sqrt(np.square(img_temp-img_masked_mean)/img_masked_var)
        return img

    def create_fin_mask(self,img):
        if img.ndim == 4:
            return np.tile(np.expand_dim(self.mask, 3), [1, 1, 1, img.shape[3]])
        img_masked = ma.masked_array(img,mask=self.mask)
        values_perc = scipy.stats.mstats.mquantiles(img_masked.flatten(),[self.perc_threshold,1-self.perc_threshold])
        mask = np.copy(self.mask)
        mask[img_masked>np.max(values_perc)] = 1
        mask[img_masked<np.min(values_perc)] = 1
        print(np.count_nonzero(mask), np.count_nonzero(self.mask), values_perc)
        return mask




