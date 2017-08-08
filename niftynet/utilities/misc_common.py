# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import datetime
import os
from functools import partial

import numpy as np
import tensorflow as tf
from scipy import ndimage

LABEL_STRINGS = ['Label', 'LABEL', 'label']


def average_grads(tower_grads):
    '''
    Performs and return the average of the gradients calculated from multiple GPUS
    :param tower_grads:
    :return ave_grads:
    '''
    # average gradients computed from multiple GPUs
    ave_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        ave_grads.append(grad_and_var)
    return ave_grads


def list_files(img_dir, ext='.nii.gz'):
    '''

    :param img_dir:
    :param ext:
    :return names: list of filenames in the list of img_dir
    that have the ext extension
    '''
    names = [fn for fn in os.listdir(img_dir) if fn.lower().endswith(ext)]
    if not names:
        print('no files in {}'.format(img_dir))
        raise IOError
    return names


def has_bad_inputs(args):
    '''
    Check if all input params have been properly set in the configuration file.
    :param args:
    :return:
    '''
    print('Input params:')
    is_bad = False
    for arg in vars(args):
        user_value = getattr(args, arg)
        if user_value is None:
            print('{} not set in the config file'.format(arg))
            is_bad = True
    ## at each iteration [batch_size] samples will be read from queue
    # if args.queue_length < args.batch_size:
    #    print('queue_length ({}) should be >= batch_size ({}).'.format(
    #        args.queue_length, args.batch_size))
    #    return True
    return is_bad


def print_save_input_parameters(args, ini_file=None):
    output_config = ['Input params at ' + str(datetime.datetime.now())[:-6]]

    for arg in vars(args):
        out_str = "-- {}: {}".format(arg, getattr(args, arg))
        print(out_str)
        output_config.append(out_str)

    if ini_file:
        with open(ini_file, 'w') as f:
            [f.write(s + '\n') for s in output_config]


class MorphologyOps(object):
    '''
    Class that performs the morphological operations needed to get notably
    connected component. To be used in the evaluation
    '''

    def __init__(self, binary_img, neigh):
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.neigh = neigh

    def border_map(self):
        '''
        Creates the border for a 3D image
        :return:
        '''
        west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
        east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
        north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
        south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
        top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
        bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
        cumulative = west + east + north + south + top + bottom
        border = ((cumulative < 6) * self.binary_map) == 1
        return border

    def foreground_component(self):
        return ndimage.label(self.binary_map)


class CacheFunctionOutput(object):
    """
    this provides a decorator to cache function outputs
    to avoid repeating some heavy function computations
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, _=None):
        if obj is None:
            return self
        return partial(self, obj)  # to remember func as self.func

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            value = cache[key]
        except KeyError:
            value = cache[key] = self.func(*args, **kw)
        return value


def look_up_operations(type_str, supported):
    assert isinstance(type_str, str) or isinstance(type_str, unicode)
    if type_str in supported and isinstance(supported, dict):
        return supported[type_str]

    if type_str in supported and isinstance(supported, set):
        return type_str

    if isinstance(supported, set):
        set_to_check = supported
    elif isinstance(supported, dict):
        set_to_check = set(supported.keys())
    else:
        set_to_check = set()

    edit_distances = {}
    for supported_key in set_to_check:
        edit_distance = _damerau_levenshtein_distance(supported_key,
                                                      type_str)
        if edit_distance <= 3:
            edit_distances[supported_key] = edit_distance
    if edit_distances:
        guess_at_correct_spelling = min(edit_distances,
                                        key=edit_distances.get)
        raise ValueError('By "{0}", did you mean "{1}"?\n '
                         '"{0}" is not a valid option.'.format(
            type_str, guess_at_correct_spelling))
    else:
        raise ValueError("no supported operation \"{}\" "
                         "is not found.".format(type_str))


def _damerau_levenshtein_distance(s1, s2):
    """Calculates an edit distance, for typo detection. Code based on :
    https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance"""
    d = {}
    string_1_length = len(s1)
    string_2_length = len(s2)
    for i in range(-1, string_1_length + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, string_2_length + 1):
        d[(-1, j)] = j + 1

    for i in range(string_1_length):
        for j in range(string_2_length):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)],
                                d[i - 2, j - 2] + cost)  # transposition

    return d[string_1_length - 1, string_2_length - 1]


def otsu_threshold(img, nbins=256):
    ''' Implementation of otsu thresholding'''
    hist, bin_edges = np.histogram(img.ravel(), bins=nbins)
    hist = hist.astype(float)
    bin_size = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_size / 2

    weight_1 = np.copy(hist)
    mean_1 = np.copy(hist)
    weight_2 = np.copy(hist)
    mean_2 = np.copy(hist)
    for i in range(1, hist.shape[0]):
        weight_1[i] = weight_1[i - 1] + hist[i]
        mean_1[i] = mean_1[i - 1] + hist[i] * bin_centers[i]

        weight_2[-i - 1] = weight_2[-i] + hist[-i - 1]
        mean_2[-i - 1] = mean_2[-i] + hist[-i - 1] * bin_centers[-i - 1]

    target_max = 0
    threshold = bin_centers[0]
    for i in range(0, hist.shape[0] - 1):
        target = weight_1[i] * weight_2[i + 1] * \
            (mean_1[i] / weight_1[i] - mean_2[i + 1] / weight_2[i + 1]) ** 2
        if target > target_max:
            target_max, threshold = target, bin_centers[i]
    return threshold


# def otsu_threshold(img, nbins=256):
#     ''' Implementation of otsu thresholding'''
#     hist, bin_edges = np.histogram(img.ravel(), bins=nbins, density=True)
#     hist = hist.astype(float) * (bin_edges[1] - bin_edges[0])
#     centre_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#
#     hist_mul_val = hist * centre_bins
#     sum_tot = np.sum(hist_mul_val)
#
#     threshold, target_max = centre_bins[0], 0
#     sum_im, mean_im = 0, 0
#     for i in range(0, hist.shape[0]-1):
#         mean_im = mean_im + hist_mul_val[i]
#         mean_ip = sum_tot - mean_im
#
#         sum_im = sum_im + hist[i]
#         sum_ip = 1 - sum_im
#
#         target = sum_ip * sum_im * np.square(mean_ip/sum_ip - mean_im/sum_im)
#         if target > target_max:
#             threshold, target_max = centre_bins[i], target
#     return threshold


# Print iterations progress
def printProgressBar(iteration, total,
                     prefix='', suffix='', decimals=1, length=10, fill='='):
    """
    Call in a loop to create terminal progress bar - To be used when
    performing the initial histogram normalisation
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
