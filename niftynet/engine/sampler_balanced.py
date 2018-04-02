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
        p(v) = (1)/(# of unique labels) * (1) / (# of voxels with same class as v)

    This is done for balanced sampling. In the case of unbalanced labels, this sampler
    should produce a roughly equal probability of sampling each class.

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
        self.spatial_coordinates_generator = balanced_spatial_coordinates


def balanced_spatial_coordinates(subject_id,
                                 data,
                                 img_sizes,
                                 win_sizes,
                                 n_samples=1):
    """
    This is the function that does the balanced sampling.

    also, note that win_sizes could be different (for example in
    a segmentation networt, theinput image window size may be 32x32x10
    and the training label window size 16x16x10 -- the network reduces
    x-y plane spatial resolution).

    This function handles this situation by first find the largest
    window across these window definitions, and generate the coordinates.
    These coordinates are then adjusted for each of the
    smaller window sizes (the output windows are concentric).
    """
    # requiring a data['sampler'] as the frequency map.
    # the shape should be [x, y, z, 1, 1]
    if data is None or data.get('sampler', None) is None:
        tf.logging.fatal("input weight map not found. please check "
                         "the configuration file")
        raise RuntimeError
    n_samples = max(n_samples, 1)
    uniq_spatial_size = set([img_size[:N_SPATIAL]
                             for img_size in list(img_sizes.values())])
    if len(uniq_spatial_size) > 1:
        tf.logging.fatal("Don't know how to generate sampling "
                         "locations: Spatial dimensions of the "
                         "grouped input sources are not "
                         "consistent. %s", uniq_spatial_size)
        raise NotImplementedError
    uniq_spatial_size = uniq_spatial_size.pop()

    # find spatial window location based on the largest spatial window
    spatial_win_sizes = [win_size[:N_SPATIAL]
                         for win_size in win_sizes.values()]
    spatial_win_sizes = np.asarray(spatial_win_sizes, dtype=np.int32)
    max_spatial_win = np.max(spatial_win_sizes, axis=0)

    # testing window size
    for i in range(0, N_SPATIAL):
        assert uniq_spatial_size[i] >= max_spatial_win[i], \
            "window size {} is larger than image size {}".format(
                max_spatial_win[i], uniq_spatial_size[i])

    # get cropped version of the input weight map where the centre of
    # the window might be. If the centre of the window was outside of
    # this crop area, the patch would be outside of the field of view
    half_win = np.floor(max_spatial_win / 2).astype(int)
    try:
        cropped_map = data['sampler'][
            half_win[0]:-half_win[0] if max_spatial_win[0] > 1 else 1,
            half_win[1]:-half_win[1] if max_spatial_win[1] > 1 else 1,
            half_win[2]:-half_win[2] if max_spatial_win[2] > 1 else 1,
            0, 0]
        assert np.all(cropped_map.shape) > 0
    except (IndexError, KeyError):
        tf.logging.fatal("incompatible map: %s", data['sampler'].shape)
        raise
    except AssertionError:
        tf.logging.fatal(
            "incompatible window size for balanced sampler. "
            "Please use smaller (fully-specified) spatial window sizes")
        raise

    # Find the number of unique labels
    flatten_map = cropped_map.flatten()
    unique_labels = np.unique(flatten_map)

    # Sample uniformly from the unique labels. This returns which labels
    # were sampled (sampled_labels) and the count of sampling for each (label_counts)
    samples = np.random.choice(unique_labels, n_samples, replace=True)
    sampled_labels, label_counts = np.unique(samples, return_counts=True)

    # Look inside each label and sample `count`. Add the middle_coord of each sample
    # to `middle_coords`
    middle_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    sample_count = 0
    for label, count in zip(sampled_labels, label_counts):
        # Get indicies where(cropped_map == label)
        valid_locations = np.where(flatten_map == label)[0]

        # Sample `count` from those indicies. Need replace=True. Consider the case where all pixels are
        # background except for one pixel which is foreground. We ask for 10 samples. We should get 5 samples
        # from background and the foreground pixel sampled 5 times (give or take random fluctuation).
        try:
            samples = np.random.choice(valid_locations, size=count, replace=True)
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

    assert sample_count == n_samples, "Internal error generating samples"

    # adjust max spatial coordinates based on each mod spatial window size
    all_coordinates = {}
    for mod in list(win_sizes):
        win_size = win_sizes[mod][:N_SPATIAL]
        half_win_diff = np.floor((max_spatial_win - win_size) / 2.0)

        # shift starting coordinates of the window
        # Note that we did not shift the centre coordinates
        # above to the corner of the window
        # because the shift is the same as the cropping amount
        # Also, we need to add half_win_diff/2 so that smaller windows
        # are centred within the large windows
        spatial_coords = np.zeros((n_samples, N_SPATIAL * 2), dtype=np.int32)
        spatial_coords[:, :N_SPATIAL] = \
            middle_coords[:, :N_SPATIAL] + half_win_diff[:N_SPATIAL]

        # the opposite corner of the window is
        # just adding the mod specific window size
        spatial_coords[:, N_SPATIAL:] = \
            spatial_coords[:, :N_SPATIAL] + win_size[:N_SPATIAL]
        # include the subject id
        subject_id = np.ones((n_samples,), dtype=np.int32) * subject_id
        spatial_coords = np.append(subject_id[:, None], spatial_coords, axis=1)
        all_coordinates[mod] = spatial_coords

    return all_coordinates
