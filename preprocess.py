import os
import pickle
import numpy as np
from external.intensity_range_standardization import\
    IntensityRangeStandardization

class HistNormaliser(object):
    def __init__(self):
        self.irs_model = []
        self.__init_precomputed_model()

    def __init_precomputed_model(self):
        self.irs_model = IntensityRangeStandardization()
        f_name = os.path.join(os.path.dirname(__file__),
                              'external/std_hist_ori_995.pkl')
        with open(f_name, 'r') as hist_ref:
            self.irs_model = pickle.load(hist_ref)
            print "reference histogram loaded"

    def intensity_normalisation(self, img, randomised=False):
        bin_id = np.random.randint(0, 20) if randomised else -1
        img = self.irs_model.transform(img, thr=bin_id)
        img = (img - np.mean(img)) / np.std(img)
        return img
