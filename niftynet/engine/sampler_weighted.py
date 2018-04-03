# -*- coding: utf-8 -*-
"""
Generating image window by weighted sampling map from input image
This can also be considered as a "weighted random cropping" layer of the
input image.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import N_SPATIAL
from niftynet.engine.sampler_uniform import UniformSampler


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
                 data_param,
                 batch_size,
                 windows_per_image,
                 queue_length=10):
        UniformSampler.__init__(self,
                                reader=reader,
                                data_param=data_param,
                                batch_size=batch_size,
                                windows_per_image=windows_per_image,
                                queue_length=queue_length)
        tf.logging.info('Initialised weighted sampler window instance')
        self.intensity_based_sampling = True
        self.middle_coordinate_sampler = weighted_spatial_coordinates


def weighted_spatial_coordinates(cropped_map, n_samples):
    """
    Weighted sampling from a map.

    This function uses a cumulative histogram for fast sampling.
    """
    # Get the cumulative sum of the normalised sorted intensities
    # i.e. first sort the sampling frequencies, normalise them
    # to sum to one, and then accumulate them in order
    flatten_map = cropped_map.flatten()
    flatten_map = flatten_map - np.min(flatten_map)
    sorted_data = np.cumsum(np.divide(np.sort(flatten_map), flatten_map.sum()))
    # get the sorting indexes to that we can invert the sorting later on.
    sorted_indexes = np.argsort(flatten_map)

    middle_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    for sample in range(0, n_samples):
        # get n_sample from the cumulative histogram, spaced by 1/n_samples,
        # plus a random perturbation to give us a stochastic sampler
        sample_ratio = 1 - (np.random.random() + sample) / (n_samples + 1)
        # find the index where the cumulative it above the sample threshold
        #     import pdb; pdb.set_trace()
        try:
            sample_index = np.argmax(sorted_data >= sample_ratio)
        except ValueError:
            tf.logging.fatal("unable to choose sampling window based on "
                             "the current frequency map.")
            raise
        # invert the sample index to the pre-sorted index
        inverted_sample_index = sorted_indexes[sample_index]
        # get the x,y,z coordinates on the cropped_map
        # (note: we need to re-shift it later due to the crop)
        middle_coords[sample, :N_SPATIAL] = np.unravel_index(
            inverted_sample_index, cropped_map.shape)[:N_SPATIAL]

    return middle_coords
