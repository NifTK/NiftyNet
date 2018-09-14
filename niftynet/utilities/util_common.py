# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import re
from functools import partial

import numpy as np
import tensorflow as tf
from scipy import ndimage
from six import string_types


def traverse_nested(input_lists, types=(list, tuple)):
    """
    Flatten a nested list or tuple
    """

    if isinstance(input_lists, types):
        for input_list in input_lists:
            for sub_list in traverse_nested(input_list, types=types):
                yield sub_list
    else:
        yield input_lists


def list_depth_count(input_list):
    """
    This function count the maximum depth of a nested list (recursively)
    This is used to check compatibility of users' input and system API
    only to be used for list or tuple
    """
    if not isinstance(input_list, (list, tuple)):
        return 0
    if len(input_list) == 0:
        return 1
    return 1 + max(map(list_depth_count, input_list))


def average_gradients(multi_device_gradients):
    # the input gradients are grouped by device,
    # this function average the gradients of multiple devices
    if multi_device_gradients is None or not multi_device_gradients:
        # nothing to average
        return multi_device_gradients

    if len(multi_device_gradients) == 1:
        # only one device, so we get rid of the first level list
        # that loops over devices
        return multi_device_gradients[0]

    nested_grads_depth = list_depth_count(multi_device_gradients)
    if nested_grads_depth == 4:
        gradients = zip(*multi_device_gradients)
        averaged_grads = [__average_grads(g) for g in gradients]
    elif nested_grads_depth == 3:
        averaged_grads = __average_grads(multi_device_gradients)
    else:
        tf.logging.fatal(
            "The list of gradients are nested in an unusual way."
            "application's gradient is not compatible with app driver."
            "Please check the return value of gradients_collector "
            "in _connect_data_and_network() of the application")
        raise RuntimeError
    return averaged_grads


def __average_grads(tower_grads):
    """
    Performs and return the average of the gradients
    :param tower_grads: in form of [[tower_1_grad], [tower_2_grad], ...]
    :return ave_grads: in form of [ave_grad]
    """
    # average gradients computed from multiple GPUs
    ave_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [tf.expand_dims(g, 0)
                 for g, _ in grad_and_vars if g is not None]
        if not grads:
            continue
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0, name='AveOverDevices')

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        ave_grads.append(grad_and_var)
    return ave_grads


def has_bad_inputs(input_args):
    """
    Check if all input params have been properly set in the configuration file.
    :param input_args:
    :return:
    """
    is_bad = False
    for section in input_args:
        section_args = input_args[section]
        for input_arg in vars(section_args):
            user_value = getattr(section_args, input_arg)
            if user_value is None:
                print('{} not set in section [{}] the config file'.format(
                    input_arg, section))
                is_bad = True

    return is_bad


def __print_argparse_section(args, section):
    output_string = []
    header_str = '[{}]'.format(section.upper())
    print(header_str)
    output_string.append(header_str)
    section_args = args[section]
    for arg in vars(section_args):
        out_str = "-- {}: {}".format(arg, getattr(section_args, arg))
        print(out_str)
        output_string.append(out_str)
    return output_string


def print_save_input_parameters(args, txt_file=None):
    import niftynet.utilities.user_parameters_parser as param_parser
    output_config = ['Input params at ' + str(datetime.datetime.now())[:-6]]
    for section in args:
        if section not in param_parser.SYSTEM_SECTIONS:
            output_config.extend(__print_argparse_section(args, section))
    for section in args:
        if section in param_parser.SYSTEM_SECTIONS:
            output_config.extend(__print_argparse_section(args, section))

    if txt_file is not None:
        with open(txt_file, 'w') as f:
            [f.write(s + '\n') for s in output_config]


class MorphologyOps(object):
    """
    Class that performs the morphological operations needed to get notably
    connected component. To be used in the evaluation
    """

    def __init__(self, binary_img, neigh):
        assert len(binary_img.shape) == 3, 'currently supports 3d inputs only'
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.neigh = neigh

    def border_map(self):
        """
        Creates the border for a 3D image
        :return:
        """
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


cache = {}


def CachedFunction(func):
    def decorated(*args, **kwargs):
        key = (func, args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return decorated


def CachedFunctionByID(func):
    def decorated(*args, **kwargs):
        id_args = tuple(id(a) for a in args)
        id_kwargs = ((k, id(kwargs[k])) for k in sorted(kwargs.keys()))
        key = (func, id_args, id_kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return decorated


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
    """
    This function validates the ``type_str`` against the supported set.

    if ``supported`` is a ``set``, returns ``type_str``
    if ``supported`` is a ``dict``, return ``supported[type_str]``
    else:
    raise an error possibly with a guess of the closest match.

    :param type_str:
    :param supported:
    :return:
    """
    assert isinstance(type_str, string_types), 'unrecognised type string'
    if isinstance(supported, dict) and type_str in supported:
        return supported[type_str]

    if isinstance(supported, set) and type_str in supported:
        return type_str

    try:
        set_to_check = set(supported)
    except TypeError:
        set_to_check = set()

    edit_distances = {}
    for supported_key in set_to_check:
        edit_distance = damerau_levenshtein_distance(supported_key,
                                                     type_str)
        if edit_distance <= 3:
            edit_distances[supported_key] = edit_distance
    if edit_distances:
        guess_at_correct_spelling = min(edit_distances,
                                        key=edit_distances.get)
        raise ValueError('By "{0}", did you mean "{1}"?\n'
                         '"{0}" is not a valid option.\n'
                         'Available options are {2}\n'.format(
                             type_str, guess_at_correct_spelling, supported))
    else:
        raise ValueError("No supported option \"{}\" "
                         "is not found.\nAvailable options are {}\n".format(
                             type_str, supported))


def damerau_levenshtein_distance(s1, s2):
    """
    Calculates an edit distance, for typo detection. Code based on :
    https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
    """
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
    """
    Implementation of otsu thresholding

    :param img:
    :param nbins:
    :return:
    """
    hist, bin_edges = np.histogram(img.ravel(), bins=nbins)
    hist = hist.astype(float)
    half_bin_size = (bin_edges[1] - bin_edges[0]) * 0.5
    bin_centers = bin_edges[:-1] + half_bin_size

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
        ratio_1 = mean_1[i] / weight_1[i]
        ratio_2 = mean_2[i + 1] / weight_2[i + 1]
        target = weight_1[i] * weight_2[i + 1] * (ratio_1 - ratio_2) ** 2
        if target > target_max:
            target_max, threshold = target, bin_centers[i]
    return threshold


# def otsu_threshold(img, nbins=256):
#     """ Implementation of otsu thresholding """
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
def print_progress_bar(iteration, total,
                       prefix='', suffix='', decimals=1, length=10, fill='='):
    """
    Call in a loop to create terminal progress bar

    :param iteration: current iteration (Int)
    :param total: total iterations (Int)
    :param prefix: prefix string (Str)
    :param suffix: suffix string (Str)
    :param decimals: number of decimals in percent complete (Int)
    :param length: character length of bar (Int)
    :param fill: bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bars = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bars, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print('\n')


def set_cuda_device(cuda_devices):
    if re.findall("\\d", cuda_devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        tf.logging.info(
            "set CUDA_VISIBLE_DEVICES to {}".format(cuda_devices))
    else:
        # using Tensorflow default choice
        pass


class ParserNamespace(object):
    """
    Parser namespace for representing parsed parameters from config file

    e.g.::

        system_params = ParserNamespace(action='train')
        action_str = system_params.action

    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)


def device_string(n_devices=0, device_id=0, is_worker=True, is_training=True):
    """
    assigning CPU/GPU based on user specifications
    """
    # pylint: disable=no-name-in-module
    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    n_local_gpus = sum([x.device_type == 'GPU' for x in devices])
    if n_devices <= 0:  # user specified no gpu at all
        return '/cpu:{}'.format(device_id)
    if is_training:
        # in training: use gpu only for workers whenever n_local_gpus
        device = 'gpu' if (is_worker and n_local_gpus > 0) else 'cpu'
        if device == 'gpu' and device_id >= n_local_gpus:
            tf.logging.warning(
                'trying to use gpu id %s, but only has %s GPU(s), '
                'please set num_gpus to %s at most',
                device_id, n_local_gpus, n_local_gpus)
            # raise ValueError
        return '/{}:{}'.format(device, device_id)
    # in inference: use gpu for everything whenever n_local_gpus
    return '/gpu:0' if n_local_gpus > 0 else '/cpu:0'


def tf_config():
    """
    tensorflow system configurations
    """
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.allow_soft_placement = True
    return config


