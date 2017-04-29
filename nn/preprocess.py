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
