# -*- coding: utf-8 -*-
"""
Generate image windows from a balanced sampling map as if every label
had the same probability of occurrence.

Consider a mask with three classes I, J, K with prevalence 0.1, 0.1, and
0.8, respectively. If 100 samples are drawn from the balanced sampler, the
classes should be approximately 33 I, 33 J, and 33 K.

This can also be considered a "balanced random cropping" layer of the
input image.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.contrib.csv_reader.sampler_uniform_v2_csv import UniformSamplerCSV
from niftynet.engine.image_window import N_SPATIAL
from niftynet.engine.sampler_weighted_v2 import crop_sampling_map


class BalancedSamplerCSV(UniformSampler):
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
                 csv_reader,
                 window_sizes,
                 batch_size=1,
                 windows_per_image=1,
                 queue_length=10,
                 name='balanced_sampler'):
        UniformSamplerCSV.__init__(self,
                                   reader=reader,
                                   csv_reader=csv_reader,
                                   window_sizes=window_sizes,
                                   batch_size=batch_size,
                                   windows_per_image=windows_per_image,
                                   queue_length=queue_length,
                                   name=name)
        tf.logging.info('Initialised balanced sampler window instance')
        self.window_centers_sampler = balanced_spatial_coordinates


def balanced_spatial_coordinates(
        n_samples, img_spatial_size, win_spatial_size, sampler_map):
    """
    Perform balanced sampling.

    Each label in the input tensor has an equal probability of
    being sampled.

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
    assert np.all(img_spatial_size[:N_SPATIAL] ==
                  sampler_map.shape[:N_SPATIAL]), \
        'image and sampling map shapes do not match'

    # Find the number of unique labels
    win_spatial_size = np.asarray(win_spatial_size, dtype=np.int32)
    cropped_map = crop_sampling_map(sampler_map, win_spatial_size)

    flatten_map = cropped_map.flatten()
    unique_labels = np.unique(flatten_map)
    if len(unique_labels) > 500:
        tf.logging.warning(
            "unusual discrete volume: number of unique "
            "labels: %s", len(unique_labels))

    # system parameter?
    class_probs = [1.0 / len(unique_labels)] * len(unique_labels)
    label_counts = np.random.multinomial(n_samples, class_probs)
    # Look inside each label and sample `count`. Add the middle_coord of
    # each sample to `middle_coords`
    middle_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    sample_count = 0
    for label, count in zip(unique_labels, label_counts):
        # Get indices where(cropped_map == label)
        valid_locations = np.where(flatten_map == label)[0]

        # Sample `count` from those indices. Need replace=True. Consider the
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
        for sample in samples:
            middle_coords[sample_count, :N_SPATIAL] = \
                np.unravel_index(sample, cropped_map.shape)[:N_SPATIAL]
            sample_count += 1

    # re-shift coords due to the crop
    half_win = np.floor(win_spatial_size / 2).astype(np.int32)
    middle_coords[:, :N_SPATIAL] = \
        middle_coords[:, :N_SPATIAL] + half_win[:N_SPATIAL]
    return middle_coords
