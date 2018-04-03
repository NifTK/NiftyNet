# -*- coding: utf-8 -*-
"""
Generate image windows from a balanced sampling map as if every label
had the same probability of occurance.

Consider a mask with three classes I, J, K with prevalence 0.1, 0.1, and
0.8, respectively. If 100 samples are drawen from the balanced sampler, the
classes should be approximately 33 I, 33 J, and 33 K.

This can also be considered a "balanced random cropping" layer of the
input image.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import N_SPATIAL
from niftynet.engine.sampler_uniform import UniformSampler


class BalancedSampler(UniformSampler):
    """
    This class generators samples from a user provided frequency map for each
    input volume. The sampling likelihood of each voxel is proportional its
    intra class frequency. That is, if a given voxel is of class `A` and there
    are 20 voxels with class `A`, the probability of selecting this voxel is
    5%. If there are 10 classes, the probability becomes 10% * 5% = 0.5%.

    In general, the likelihood of sampling a voxel is given by:
        p(v) = (1)/(# of unique labels * # of voxels with same class as v)

    This is done for balanced sampling. In the case of unbalanced labels,
    this sampler should produce a roughly equal probability of sampling each
    class.

    This layer can be considered as a "balanced random cropping" layer of the
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
        tf.logging.info('Initialised balanced sampler window instance')
        self.intensity_based_sampling = True
        self.middle_coordinate_sampler = balanced_spatial_coordinates


def balanced_spatial_coordinates(cropped_map, n_samples):
    """
    Perform balanced sampling.

    Each label in the input tensor has an equal probability of
    being sampled.
    """
    # Find the number of unique labels
    flatten_map = cropped_map.flatten()
    unique_labels = np.unique(flatten_map)

    # Sample uniformly from the unique labels. This returns which labels
    # were sampled (sampled_labels) and the count of sampling for each
    # (label_counts)
    samples = np.random.choice(unique_labels, n_samples, replace=True)
    sampled_labels, label_counts = np.unique(samples, return_counts=True)

    # Look inside each label and sample `count`. Add the middle_coord of
    # each sample to `middle_coords`
    middle_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    sample_count = 0
    for label, count in zip(sampled_labels, label_counts):
        # Get indicies where(cropped_map == label)
        valid_locations = np.where(flatten_map == label)[0]

        # Sample `count` from those indicies. Need replace=True. Consider the
        # case where all pixels are background except for one pixel which is
        # foreground. We ask for 10 samples. We should get 5 samples from
        # background and the foreground pixel sampled 5 times (give or take
        # random fluctuation).
        try:
            samples = np.random.choice(
                valid_locations,
                size=count,
                replace=True)
        except ValueError:
            tf.logging.fatal("unable to choose sampling window based on "
                             "the current frequency map.")
            raise

        assert count == samples.size, "Unable to sample from the image"

        # Place into `middle_coords`
        for i in range(0, count):
            middle_coords[sample_count, :N_SPATIAL] = \
                np.unravel_index(samples[i], cropped_map.shape)[:N_SPATIAL]
            sample_count += 1

    return middle_coords
