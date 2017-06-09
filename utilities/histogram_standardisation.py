# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os

import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndimg
from scipy.ndimage.morphology import binary_fill_holes as fill_holes

import utilities.misc_io as io
from utilities.misc_common import look_up_operations, printProgressBar

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
"""
Implementation of
Nyúl László G., Jayaram K. Udupa, and Xuan Zhang.
"New variants of a method of MRI scale standardization."
IEEE transactions on medical imaging 19.2 (2000): 143-150.
"""

DEFAULT_CUTOFF = [0.01, 0.99]
SUPPORTED_MASK_TYPES = {'threshold_plus', 'threshold_minus',
                        'otsu_plus', 'otsu_minus', 'mean'}
SUPPORTED_CUTPOINTS = {'percentile', 'quartile', 'median'}


def __compute_percentiles(img, mask, cutoff):
    perc = [cutoff[0],
            0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9,
            cutoff[1]]
    masked_img = ma.masked_array(img, np.logical_not(mask)).compressed()
    perc_results = np.percentile(masked_img, 100 * np.array(perc))
    # hist, bin = np.histogram(ma.compressed(masked_img), bins=50)
    return perc_results


def __standardise_cutoff(cutoff, type_hist='quartile'):
    cutoff = np.asarray(cutoff)
    if cutoff is None:
        return DEFAULT_CUTOFF
    if len(cutoff) > 2:
        cutoff = np.unique([np.min(cutoff), np.max(cutoff)])
    if len(cutoff) < 2:
        return DEFAULT_CUTOFF
    if cutoff[0] > cutoff[1]:
        cutoff[0], cutoff[1] = cutoff[1], cutoff[0]
    cutoff[0] = max(0., cutoff[0])
    cutoff[1] = min(1., cutoff[1])
    if type_hist == 'quartile':
        cutoff[0] = np.min([cutoff[0], 0.24])
        cutoff[1] = np.max([cutoff[1], 0.76])
    else:
        cutoff[0] = np.min([cutoff[0], 0.09])
        cutoff[1] = np.max([cutoff[1], 0.91])
    return cutoff


def create_mapping_from_multimod_arrayfiles(array_files,
                                            list_modalities,
                                            cutoff,
                                            mask_type):
    perc_database = {}
    for (i, p) in enumerate(array_files):
        printProgressBar(i, len(array_files),
                         prefix='normalisation histogram training',
                         decimals=1, length=10, fill='*')
        img_data = io.csv_cell_to_volume_5d(p)
        # numb_modalities = img_data.shape[3]
        numb_timepoints = img_data.shape[4]
        # to_do = {m: list_modalities[m] for m in list_modalities.keys() if
        #         list_modalities[m] < numb_modalities}
        for m in list_modalities.keys():
            if m not in perc_database.keys():
                perc_database[m] = []
            for t in range(0, numb_timepoints):
                img_3d = img_data[..., list_modalities[m], t]
                mask_3d = create_mask_img_3d(img_3d, mask_type)
                perc = __compute_percentiles(img_3d, mask_3d, cutoff)
                perc_database[m].append(perc)
    mapping = {}
    for m in list_modalities.keys():
        perc_database[m] = np.vstack(perc_database[m])
        s1, s2 = create_standard_range()
        mapping[m] = __averaged_mapping(perc_database[m], s1, s2)
    return mapping


def create_standard_range():
    return 0., 100.


def create_mask_img_3d(img, type_mask='otsu_plus', thr=0.):
    assert img.ndim == 3
    type_mask = look_up_operations(type_mask.lower(), SUPPORTED_MASK_TYPES)
    mask = np.zeros_like(img, dtype=np.bool)
    if type_mask == 'threshold_plus':
        mask[img > thr] = 1
    elif type_mask == 'threshold_minus':
        mask[img < thr] = 1
    elif type_mask == 'otsu_plus':
        if np.any(img):
            thr = filters.threshold_otsu(img)
        mask[img > thr] = 1
    elif type_mask == 'otsu_minus':
        if np.any(img):
            thr = filters.threshold_otsu(img)
        mask[img < thr] = 1
    elif type_mask == 'mean':
        thr = np.mean(img)
        mask[img > thr] = 1
    mask = ndimg.binary_dilation(mask, iterations=2)
    mask = fill_holes(mask)
    assert not np.all(mask==False)
    # mask_fin = ndimg.binary_erosion(mask_bis, iterations=2)
    return mask


def __averaged_mapping(perc_database, s1, s2):
    # assuming shape: n_data_points = perc_database.shape[0]
    #                 n_percentiles = perc_database.shape[1]
    slope = (s2 - s1) / (perc_database[:, -1] - perc_database[:, 0])
    slope = np.nan_to_num(slope)
    final_map = slope.dot(perc_database) / perc_database.shape[0]
    intercept = np.mean(s1 - slope * perc_database[:, 0])
    final_map = final_map + intercept
    return final_map


# TODO: test cases
def transform_by_mapping(img, mask, mapping, cutoff, type_hist='quartile'):
    type_hist = look_up_operations(type_hist.lower(), SUPPORTED_CUTPOINTS)
    if type_hist == 'quartile':
        range_to_use = [0, 3, 6, 9, 12]
    elif type_hist == 'percentile':
        range_to_use = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]
    elif type_hist == 'median':
        range_to_use = [0, 6, 12]
    else:
        raise ValueError('unknown cutting points type')
    cutoff = __standardise_cutoff(cutoff, type_hist)
    perc = __compute_percentiles(img, mask, cutoff)
    # Apply linear histogram standardisation
    range_mapping = mapping[range_to_use]
    range_perc = perc[range_to_use]
    diff_mapping = range_mapping[1:] - range_mapping[:-1]
    diff_perc = range_perc[1:] - range_perc[:-1]
    diff_perc[diff_perc == 0] = 1e-5

    affine_map = np.zeros([2, len(range_to_use) - 1])
    # compute slopes of the linear models
    affine_map[0] = diff_mapping / diff_perc
    # compute intercepts of the linear models
    affine_map[1] = range_mapping[:-1] - affine_map[0] * range_perc[:-1]
    # lin_img = np.ones_like(img, dtype=np.float32)
    # aff_img = np.zeros_like(img, dtype=np.float32)

    # img < range_perc[0] set to affine_map[default], 1, 0
    # img >= range_perc[9] set to affine_map[:,9]
    # for i in range(len(range_to_use) - 1):
    #    greater_than_i = (img >= range_perc[i])
    #    lin_img[greater_than_i] = affine_map[0, i]
    #    aff_img[greater_than_i] = affine_map[1, i]

    # img < range_perc[0] set to affine_map[:,-1]
    # img >= range_perc[9] set to affine_map[:,9]
    # by the design of np.digitize, if img >= range_perc[i]: return i+1)
    bin_id = np.digitize(img, range_perc[:-1], right=False) - 1
    lin_img = affine_map[0, bin_id]
    aff_img = affine_map[1, bin_id]
    # handling below cutoff[0] over cutoff[1]
    # values are mapped linearly and then smoothed
    lowest_values = img <= range_perc[0]
    highest_values = img >= range_perc[-1]
    lin_img[lowest_values] = affine_map[0, 0]
    aff_img[lowest_values] = affine_map[1, 0]
    new_img = lin_img * img + aff_img
    # Apply smooth thresholding (exponential) below cutoff[0] and over cutoff[1]
    new_img[lowest_values] = smooth_threshold(
        new_img[lowest_values], mode='low')
    new_img[highest_values] = smooth_threshold(
        new_img[highest_values], mode='high')
    # Apply mask and set background to zero
    new_img[mask == False] = 0.
    return new_img


def smooth_threshold(value, mode='high'):
    smoothness = 1.
    if mode == 'high':
        affine = np.min(value)
        smooth_value = (value - affine) / smoothness
        smooth_value = (1. - np.exp((-1) * smooth_value)) + affine
    elif mode == 'low':
        affine = np.max(value)
        smooth_value = (value - affine) / smoothness
        smooth_value = (np.exp(smooth_value) - 1.) + affine
    else:
        smooth_value = value
    return smooth_value


def read_mapping_file(mapping_file):
    mapping_dict = {}
    if not os.path.isfile(mapping_file):
        return mapping_dict
    with open(mapping_file, "r") as f:
        for line in f:
            if len(line) <= 2:
                continue
            line = line.split()
            if len(line) < 2:
                continue
            map_name, map_value = line[0], np.float32(line[1:])
            mapping_dict[map_name] = map_value
    return mapping_dict


def force_writing_new_mapping(filename, mapping_dict):
    f = open(filename, 'w+')
    for mod in mapping_dict.keys():
        mapping_string = ' '.join(map(str, mapping_dict[mod]))
        string_fin = '{} {}\n'.format(mod, mapping_string)
        f.write(string_fin)
