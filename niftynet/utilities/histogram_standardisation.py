# -*- coding: utf-8 -*-
"""
Implementation of
Nyúl László G., Jayaram K. Udupa, and Xuan Zhang.
"New variants of a method of MRI scale standardization."
IEEE transactions on medical imaging 19.2 (2000): 143-150.

This implementation only supports input images with floating point number,
(not integers).
"""
from __future__ import absolute_import, print_function, division

import os

import numpy as np
import numpy.ma as ma
import tensorflow as tf

from niftynet.io.misc_io import touch_folder
from niftynet.utilities.util_common import \
    look_up_operations, print_progress_bar

DEFAULT_CUTOFF = [0.01, 0.99]
SUPPORTED_CUTPOINTS = set(['percentile', 'quartile', 'median'])


def __compute_percentiles(img, mask, cutoff):
    """
    Creates the list of percentile values to be used as landmarks for the
    linear fitting.

    :param img: Image on which to determine the percentiles
    :param mask: Mask to use over the image to constraint to the relevant
    information
    :param cutoff: Values of the minimum and maximum percentiles to use for
    the linear fitting
    :return perc_results: list of percentiles value for the given image over
    the mask
    """
    perc = [cutoff[0],
            0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9,
            cutoff[1]]
    masked_img = ma.masked_array(img, np.logical_not(mask)).compressed()
    perc_results = np.percentile(masked_img, 100 * np.array(perc))
    # hist, bin = np.histogram(ma.compressed(masked_img), bins=50)
    return perc_results


def __standardise_cutoff(cutoff, type_hist='quartile'):
    """
    Standardises the cutoff values given in the configuration

    :param cutoff:
    :param type_hist: Type of landmark normalisation chosen (median,
    quartile, percentile)
    :return cutoff: cutoff with appropriate adapted values
    """
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
                                            field,
                                            modalities,
                                            mod_to_train,
                                            cutoff,
                                            masking_function):
    """
    Performs the mapping creation based on a list of files. For each of the
    files (potentially multimodal), the landmarks are defined for each
    modality and stored in a database. The average of these landmarks is
    returned providing the landmarks to use for the linear mapping of any
    new incoming data

    :param array_files: List of image files to use
    :param modalities: Name of the modalities used for the
        standardisation and the corresponding order in the multimodal files
    :param cutoff: Minimum and maximum landmarks percentile values to use for
        the mapping
    :param masking_function: Describes how the mask is defined for each image.
    :return:
    """
    perc_database = {}
    for (i, p) in enumerate(array_files):
        print_progress_bar(i, len(array_files),
                           prefix='normalisation histogram training',
                           decimals=1, length=10, fill='*')
        img_data = p[field].get_data()
        assert img_data.shape[4] == len(modalities), \
            "number of modalities are not consistent in the input image"
        for mod_i, m in enumerate(modalities):
            if m not in mod_to_train:
                continue
            if m not in perc_database.keys():
                perc_database[m] = []
            for t in range(img_data.shape[3]):
                img_3d = img_data[..., t, mod_i]
                if masking_function is not None:
                    mask_3d = masking_function(img_3d)
                else:
                    mask_3d = np.ones_like(img_3d, dtype=np.bool)
                perc = __compute_percentiles(img_3d, mask_3d, cutoff)
                perc_database[m].append(perc)
    mapping = {}
    for m in list(perc_database):
        perc_database[m] = np.vstack(perc_database[m])
        s1, s2 = create_standard_range()
        mapping[m] = tuple(__averaged_mapping(perc_database[m], s1, s2))
    return mapping


def create_standard_range():
    return 0., 100.


def __averaged_mapping(perc_database, s1, s2):
    """
    Map the landmarks of the database to the chosen range
    :param perc_database: perc_database over which to perform the averaging
    :param s1, s2: limits of the mapping range
    :return final_map: the average mapping
    """
    # assuming shape: n_data_points = perc_database.shape[0]
    #                 n_percentiles = perc_database.shape[1]
    slope = (s2 - s1) / (perc_database[:, -1] - perc_database[:, 0])
    slope = np.nan_to_num(slope)
    final_map = slope.dot(perc_database) / perc_database.shape[0]
    intercept = np.mean(s1 - slope * perc_database[:, 0])
    final_map = final_map + intercept
    return final_map


def transform_by_mapping(img, mask, mapping, cutoff, type_hist='quartile'):
    """
    Performs the standardisation of a given image.

    :param img: image to standardise
    :param mask: mask over which to determine the landmarks
    :param mapping: mapping landmarks to use for the piecewise linear
        transformations
    :param cutoff: cutoff points for the mapping
    :param type_hist: Type of landmarks scheme to use: choice between
        quartile percentile and median
    :return new_img: the standardised image
    """
    image_shape = img.shape
    img = img.reshape(-1)
    mask = mask.reshape(-1)

    type_hist = look_up_operations(type_hist.lower(), SUPPORTED_CUTPOINTS)
    if type_hist == 'quartile':
        range_to_use = [0, 3, 6, 9, 12]
    elif type_hist == 'percentile':
        range_to_use = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]
    elif type_hist == 'median':
        range_to_use = [0, 6, 12]
    else:
        raise ValueError('unknown cutting points type_str')
    assert len(mapping) >= len(range_to_use), \
        "wrong mapping format, please check the histogram reference file"
    mapping = np.asarray(mapping)
    cutoff = __standardise_cutoff(cutoff, type_hist)
    perc = __compute_percentiles(img, mask, cutoff)
    # Apply linear histogram standardisation
    range_mapping = mapping[range_to_use]
    range_perc = perc[range_to_use]
    diff_mapping = range_mapping[1:] - range_mapping[:-1]
    diff_perc = range_perc[1:] - range_perc[:-1]

    # handling the case where two landmarks are the same
    # for a given input image.  This usually happens when
    # image background are not removed from the image.
    diff_perc[diff_perc == 0] = np.inf

    affine_map = np.zeros([2, len(range_to_use) - 1])
    # compute slopes of the linear models
    affine_map[0] = diff_mapping / diff_perc
    # compute intercepts of the linear models
    affine_map[1] = range_mapping[:-1] - affine_map[0] * range_perc[:-1]

    bin_id = np.digitize(img, range_perc[1:-1], right=False)
    lin_img = affine_map[0, bin_id]
    aff_img = affine_map[1, bin_id]
    # handling below cutoff[0] over cutoff[1]
    # values are mapped linearly and then smoothed
    new_img = lin_img * img + aff_img

    # Apply smooth thresholding (exponential)
    # below cutoff[0] and over cutoff[1]
    # this might not guarantee one to one mapping

    # lowest_values = img <= range_perc[0]
    # highest_values = img >= range_perc[-1]
    # new_img[lowest_values] = smooth_threshold(
    #     new_img[lowest_values], mode='low')
    # new_img[highest_values] = smooth_threshold(
    #     new_img[highest_values], mode='high')

    # Apply mask and set background to zero
    # new_img[mask == False] = 0.
    new_img = new_img.reshape(image_shape)
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
    """
    Reads an existing mapping file with the given modalities.
    :param mapping_file: file in which mapping is stored
    :return mapping_dict: dictionary containing the mapping landmarks for
    each modality stated in the mapping file
    """
    mapping_dict = {}
    if not mapping_file:
        return mapping_dict
    if not os.path.isfile(mapping_file):
        return mapping_dict

    with open(mapping_file, "r") as f:
        for line in f:
            if len(line) <= 2:
                continue
            line = line.split()
            if len(line) < 2:
                continue
            try:
                map_name, map_value = line[0], np.float32(line[1:])
                mapping_dict[map_name] = tuple(map_value)
            except ValueError:
                tf.logging.fatal(
                    "unknown input format: {}".format(mapping_file))
                raise
    return mapping_dict


# Function to modify the model file with the mapping if needed according
# to existent mapping and modalities
def write_all_mod_mapping(hist_model_file, mapping):
    # backup existing file first
    if os.path.exists(hist_model_file):
        backup_name = '{}.backup'.format(hist_model_file)
        from shutil import copyfile
        try:
            copyfile(hist_model_file, backup_name)
        except OSError:
            tf.logging.warning('cannot backup file {}'.format(hist_model_file))
            raise
        tf.logging.warning(
            "moved existing histogram reference file\n"
            " from {} to {}".format(hist_model_file, backup_name))

    touch_folder(os.path.dirname(hist_model_file))
    __force_writing_new_mapping(hist_model_file, mapping)


def __force_writing_new_mapping(filename, mapping_dict):
    """
    Writes a mapping dictionary to file

    :param filename: name of the file in which to write the saved mapping
    :param mapping_dict: mapping dictionary to save in the file
    :return:
    """
    with open(filename, 'w+') as f:
        for mod in mapping_dict.keys():
            mapping_string = ' '.join(map(str, mapping_dict[mod]))
            string_fin = '{} {}\n'.format(mod, mapping_string)
            f.write(string_fin)
    return
