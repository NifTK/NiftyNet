# -*- coding: utf-8 -*-
import os
import pickle
import sys

import numpy as np

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
