from __future__ import absolute_import, print_function, division

import numpy as np
import numpy.ma as ma
import math
import scipy.stats.mstats as mstats
import scipy.ndimage as ndimage

from niftynet.utilities.misc_common import MorphologyOps, CacheFunctionOutput


class RegionProperties(object):
    def __init__(self, seg, img, measures,
                 num_neighbors=24, threshold=0, pixdim=[1, 1, 1]):

        self.seg = seg
        self.img = img
        self.img_channels = self.img.shape[3] if img.ndim >= 4 else 1
        img_id = range(0, self.img_channels)
        self.m_dict = {
            'centre of mass': (self.centre_of_mass, ['CoMx',
                                                     'CoMy',
                                                     'CoMz']),
            'volume': (self.volume,
                       ['NVoxels', 'NVoxelsBinary', 'Vol', 'VolBinary']),
            'surface': (self.surface, ['NSurface',
                                       'NSurfaceBinary',
                                       'SurfaceVol',
                                       'SurfaceVolBinary']),
            'surface volume ratio': (self.sav, ['SAVNumb',
                                                'SAVNumBinary',
                                                'SAV',
                                                'SAVBinary']),
            'compactness': (self.compactness, ['CompactNumb',
                                               'CompactNumbBinary',
                                               'Compactness',
                                               'CompactnessBinary']),
            'mean': (self.mean_, ['Mean_%d' % i for i in img_id]),
            'weighted_mean': (self.weighted_mean_,
                              ['Weighted_mean_%d' % i for i in img_id]),
            'median': (self.median_, ['Median_%d' % i for i in img_id]),
            'skewness': (self.skewness_, ['Skewness_%d' % i for i in img_id]),
            'kurtosis': (self.kurtosis_, ['Kurtosis_%d' % i for i in img_id]),
            'min': (self.min_, ['Min_%d' % i for i in img_id]),
            'max': (self.max_, ['Max_%d' % i for i in img_id]),
            'quantile_25': (self.quantile_25,
                            ['P25_%d' % i for i in img_id]),
            'quantile_50': (self.median_,
                            ['P50_%d' % i for i in img_id]),
            'quantile_75': (self.quantile_75,
                            ['P75_%d' % i for i in img_id]),
            'std': (self.std_, ['STD_%d' % i for i in img_id]),
            'harilick_features': (self.harilick, [''])
        }
        self.measures = measures
        self.neigh = num_neighbors
        self.pixdim = pixdim
        self.threshold = threshold
        if self.seg is not None:
            self.masked_img, self.masked_seg = self.__compute_mask()
        self.vol_vox = np.prod(pixdim)

    def __compute_mask(self):
        # TODO: check whether this works for probabilities type
        foreground_selector = np.where((self.seg > 0).reshape(-1))[0]
        probs = self.seg.reshape(-1)[foreground_selector]
        regions = np.zeros((foreground_selector.shape[0], self.img_channels))
        for i in np.arange(self.img_channels):
            regions[:, i] = self.img[..., i, 0].reshape(-1)[foreground_selector]
        return regions, probs

    def centre_of_mass(self):
        return np.mean(np.argwhere(self.seg > self.threshold), 0)

    @CacheFunctionOutput
    def volume(self):
        numb_seg = np.sum(self.seg)
        numb_seg_bin = np.sum(self.seg > 0)
        return numb_seg, numb_seg_bin, \
               numb_seg * self.vol_vox, numb_seg_bin * self.vol_vox

    @CacheFunctionOutput
    def surface(self):
        border_seg = MorphologyOps(self.seg, self.neigh).border_map()
        numb_border_seg_bin = np.sum(border_seg > 0)
        numb_border_seg = np.sum(border_seg)
        return numb_border_seg, numb_border_seg_bin, \
               numb_border_seg * self.vol_vox, numb_border_seg_bin * self.vol_vox

    def glcm(self):
        shifts = [[0,0,0,0,0],
                  [1,0,0,0,0],
                  [-1,0,0,0,0],
                  [0,1,0,0,0],
                  [0,-1,0,0,0],
                  [0, 0, 1, 0, 0],
                  [0, 0, -1, 0, 0],
                  [1, 1, 0, 0, 0],
                  [-1, -1, 0, 0, 0],
                  [-1, 1, 0, 0, 0],
                  [1, -1, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [0, -1, -1, 0, 0],
                  [0, -1, 1, 0, 0],
                  [0, 1, -1, 0, 0],
                  [1, 0,1, 0, 0],
                  [-1,0, -1, 0, 0],
                  [-1,0, 1, 0, 0],
                  [1, 0,-1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [-1, 1, -1, 0, 0],
                  [-1, 1, 1, 0, 0],
                  [1, 1, -1, 0, 0],
                  [1, -1, 1, 0, 0],
                  [-1, -1, -1, 0, 0],
                  [-1, -1, 1, 0, 0],
                  [1, -1, -1, 0, 0]]
        bins = np.arange(0, self.bin)
        multi_mod_glcm = []
        for m in range(0, self.img.shape[4]):
            shifted_image = []
            for n in range(0, self.neigh+1):
                new_img = np.multiply(self.seg, self.img[..., m:m+1, 0:1])
                new_img = ndimage.shift(new_img, shifts[n], order=0)
                if np.count_nonzero(new_img) > 0:
                    flattened_new = np.flatten(new_img)
                    flattened_seg = np.flatten(self.seg)
                    select = [round(flattened_new[i] * self.mul+self.trans) for i in
                                    range(0, new_img.size) if
                              flattened_seg[i]>0]

                    select_new = np.digitize(select, bins)
                    shifted_image.append(select_new)
            glcm = np.zeros([self.bin, self.bin, self.neigh])
            for n in range(0, self.neigh):
                for i in range(0, shifted_image[0].size):
                    glcm[shifted_image[0][i], shifted_image[n+1][i], n] += 1
            glcm = glcm / np.sum(np.sum(glcm, axis=0), axis=1)
            multi_mod_glcm.append(glcm)
        return multi_mod_glcm

    def harilick(self):
        multi_mod_glcm = self.glcm()


    def contrast(self, matrix):
        contrast = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                contrast += (j-i)**2 * matrix[i,j]
        return contrast

    def homogeneity(self, matrix):
        homogeneity = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                homogeneity += matrix[i,j]/(1-abs(i-j))
        return homogeneity

    def energy(self, matrix):
        energy = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                energy += matrix[i,j] ** 2
        return energy

    def entropy(self, matrix):
        entropy = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                if matrix[i,j] > 0:
                    entropy += matrix[i,j] * math.log(matrix[i,j])
        return entropy


    def correlation(self, matrix):
        range_values = np.arange(0, matrix.shape[0])
        matrix_range = np.tile(range_values, [1, matrix.shape[0]])
        mean = np.average(matrix_range, weights=matrix, axis=0)
        sd = math.sqrt(np.average((matrix_range-np.tile(mean,[1,matrix.shape[
            0]]))**2, weights=matrix, axis=0))
        correlation = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                correlation += (i*j*matrix[i, j]-mean[i] * mean[j]) / (sd[i] *
                                                                   sd[j])
        return correlation

    def inverse_difference_moment(self, matrix):
        idm = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                idm += 1.0 / (1 + (i-j)**2) * matrix[i,j]
        return idm

    def sum_average(self, matrix):
        sa = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                sa += (i+j) * matrix[i,j]
        return sa

    def sum_entropy(self, matrix):
        se = 0
        matrix_bis = np.zeros([2*matrix.shape[0]])
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                matrix_bis[i+j] += matrix[i, j]
        for v in matrix_bis:
            if v > 0:
                se += v*math.log(v)
        return se



    def sum_square_variance(self, matrix):
        ssv = 0
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[0]):
                ssv += (i-mean[i]) ** 2 * matrix[i,j]


    def sav(self):
        Sn, Snb, Sv, Svb = self.surface()
        Vn, Vnb, Vv, Vvb = self.volume()
        return Sn / Vn, Snb / Vnb, Sv / Vv, Svb / Vvb

    def compactness(self):
        Sn, Snb, Sv, Svb = self.surface()
        Vn, Vnb, Vv, Vvb = self.volume()
        return np.power(Sn, 1.5) / Vn, np.power(Snb, 1.5) / Vnb, \
               np.power(Sv, 1.5) / Vv, np.power(Svb, 1.5) / Vvb

    def min_(self):
        return ma.min(self.masked_img, 0)

    def max_(self):
        return ma.max(self.masked_img, 0)

    def weighted_mean_(self):
        masked_seg = np.tile(self.masked_seg, [self.img_channels, 1]).T
        return ma.average(self.masked_img, axis=0, weights=masked_seg).flatten()

    def mean_(self):
        return ma.mean(self.masked_img, 0)

    def skewness_(self):
        return mstats.skew(self.masked_img, 0)

    def std_(self):
        return ma.std(self.masked_img, 0)

    def kurtosis_(self):
        return mstats.kurtosis(self.masked_img, 0)

    def median_(self):
        return ma.median(self.masked_img, 0)

    def quantile_25(self):
        return mstats.mquantiles(self.masked_img, prob=0.25, axis=0).flatten()

    def quantile_75(self):
        return mstats.mquantiles(self.masked_img, prob=0.75, axis=0).flatten()

    def header_str(self):
        result_str = [j for i in self.measures for j in self.m_dict[i][1]]
        result_str = ',' + ','.join(result_str)
        return result_str

    def to_string(self, fmt='{:4f}'):
        result_str = ""
        for i in self.measures:
            for j in self.m_dict[i][0]():
                try:
                    result_str += ',' + fmt.format(j)
                except ValueError:  # some functions give strings e.g., "--"
                    print(i, j)
                    result_str += ',{}'.format(j)
        return result_str
