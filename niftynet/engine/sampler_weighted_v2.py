# -*- coding: utf-8 -*-
"""
Generating image window by weighted sampling map from input image
This can also be considered as a "weighted random cropping" layer of the
input image.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.engine.image_window import N_SPATIAL


class WeightedSampler(UniformSampler):
    """
    This class generators samples from a user provided
    frequency map for each input volume
    The sampling likelihood of each voxel (and window around)
    is proportional to its frequency

    This is implemented in a closed form using cumulative histograms
    for efficiency purposes i.e., the first three dims of image.

    This layer can be considered as a "weighted random cropping" layer of the
    input image.
    """

    def __init__(self,
                 reader,
                 window_sizes,
                 batch_size=1,
                 windows_per_image=1,
                 queue_length=10,
                 name='weighted_sampler'):
        UniformSampler.__init__(self,
                                reader=reader,
                                window_sizes=window_sizes,
                                batch_size=batch_size,
                                windows_per_image=windows_per_image,
                                queue_length=queue_length,
                                name=name)
        tf.logging.info('Initialised weighted sampler window instance')
        self.window_centers_sampler = weighted_spatial_coordinates


def weighted_spatial_coordinates(
        n_samples, img_spatial_size, win_spatial_size, sampler_map):
    """
    Weighted sampling from a map.
    This function uses a cumulative histogram for fast sampling.

    see also `sampler_uniform.rand_spatial_coordinates`

    :param n_samples: number of random coordinates to generate
    :param img_spatial_size: input image size
    :param win_spatial_size: input window size
    :param sampler_map: sampling prior map, it's spatial shape should be
            consistent with `img_spatial_size`
    :return: (n_samples, N_SPATIAL) coordinates representing sampling
              window centres relative to img_spatial_size
    """
    assert sampler_map is not None, \
        'sampling prior map is not specified, ' \
        'please check `sampler=` option in the config.'
    # Get the cumulative sum of the normalised sorted intensities
    # i.e. first sort the sampling frequencies, normalise them
    # to sum to one, and then accumulate them in order
    assert np.all(img_spatial_size[:N_SPATIAL] ==
                  sampler_map.shape[:N_SPATIAL]), \
        'image and sampling map shapes do not match'
    win_spatial_size = np.asarray(win_spatial_size, dtype=np.int32)
    cropped_map = crop_sampling_map(sampler_map, win_spatial_size)
    flatten_map = cropped_map.flatten()
    flatten_map_min = np.min(flatten_map)
    if flatten_map_min < 0:
        flatten_map -= flatten_map_min
    normaliser = flatten_map.sum()
    # get the sorting indexes to that we can invert the sorting later on.
    sorted_indexes = np.argsort(flatten_map)
    sorted_data = np.cumsum(
        np.true_divide(flatten_map[sorted_indexes], normaliser))

    middle_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    for sample in range(n_samples):
        # get n_sample from the cumulative histogram, spaced by 1/n_samples,
        # plus a random perturbation to give us a stochastic sampler
        sample_ratio = 1 - (np.random.random() + sample) / n_samples
        # find the index where the cumulative it above the sample threshold
        try:
            if normaliser == 0:
                # constant map? reducing to a uniform sampling
                sample_index = np.random.randint(len(sorted_data))
            else:
                sample_index = np.argmax(sorted_data >= sample_ratio)
        except ValueError:
            tf.logging.fatal("unable to choose sampling window based on "
                             "the current frequency map.")
            raise
        # invert the sample index to the pre-sorted index
        inverted_sample_index = sorted_indexes[sample_index]
        # get the x,y,z coordinates on the cropped_map
        middle_coords[sample, :N_SPATIAL] = np.unravel_index(
            inverted_sample_index, cropped_map.shape)[:N_SPATIAL]

    # re-shift coords due to the crop
    half_win = np.floor(win_spatial_size / 2).astype(np.int32)
    middle_coords[:, :N_SPATIAL] = \
        middle_coords[:, :N_SPATIAL] + half_win[:N_SPATIAL]
    return middle_coords


def crop_sampling_map(input_map, win_spatial_size):
    """
    Utility function for generating a cropped version of the
    input sampling prior map (the input weight map where the centre of
    the window might be). If the centre of the window was outside of
    this crop area, the patch would be outside of the field of view

    :param input_map: the input weight map where the centre of
                      the window might be
    :param win_spatial_size: size of the borders to be cropped
    :return: cropped sampling map
    """

    # prepare cropping indices
    _start, _end = [], []
    for win_size, img_size in \
            zip(win_spatial_size[:N_SPATIAL], input_map.shape[:N_SPATIAL]):
        # cropping floor of the half window
        d_start = int(win_size / 2.0)
        # using ceil of half window
        d_end = img_size - win_size + int(win_size / 2.0 + 0.6)

        _start.append(d_start)
        _end.append(d_end + 1 if d_start == d_end else d_end)

    try:
        assert len(_start) == 3
        cropped_map = input_map[
            _start[0]:_end[0], _start[1]:_end[1], _start[2]:_end[2], 0, 0]
        assert np.all(cropped_map.shape) > 0
    except (IndexError, KeyError, TypeError, AssertionError):
        tf.logging.fatal(
            "incompatible map: %s and window size: %s\n"
            "try smaller (fully-specified) spatial window sizes?",
            input_map.shape, win_spatial_size)
        raise
    return cropped_map
