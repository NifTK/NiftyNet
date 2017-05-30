# -*- coding: utf-8 -*-
import os
from functools import partial
from skimage import measure

import numpy as np
import tensorflow as tf
from scipy import ndimage

LABEL_STRINGS = ['Label', 'LABEL', 'label']


def average_grads(tower_grads):
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
    names = [fn for fn in os.listdir(img_dir) if fn.lower().endswith(ext)]
    if not names:
        print('no files in {}'.format(img_dir))
        raise IOError
    return names


def has_bad_inputs(args):
    print('Input params:')
    for arg in vars(args):
        user_value = getattr(args, arg)
        print("-- {}: {}".format(arg, getattr(args, arg)))
        if user_value is None:
            print('{} not set in the config file'.format(arg))
            return True
    ## at each iteration [batch_size] samples will be read from queue
    # if args.queue_length < args.batch_size:
    #    print('queue_length ({}) should be >= batch_size ({}).'.format(
    #        args.queue_length, args.batch_size))
    #    return True
    return False


class MorphologyOps(object):
    def __init__(self, binary_img, neigh):
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.neigh = neigh

    def border_map(self):
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
        return measure.label(self.binary_map, background=0)


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


def damerau_levenshtein_distance(s1, s2):
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
