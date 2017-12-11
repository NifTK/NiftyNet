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
        self.spatial_coordinates_generator = weighted_spatial_coordinates


def weighted_spatial_coordinates(subject_id,
                                 data,
                                 img_sizes,
                                 win_sizes,
                                 n_samples=1):
    """
    This is the function that actually does the cumulative histogram
    and sampling.

    also, note that win_sizes could be different
    (for example in segmentation network
    input image window size is 32x32x10,
    training label window is 16x16x10 -- the network reduces x-y plane
    spatial resolution).

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
            "incompatible window size for weighted sampler. "
            "Please use smaller (fully-specified) spatial window sizes")
        raise
    # Get the cumulative sum of the normalised sorted intensities
    # i.e. first sort the sampling frequencies, normalise them
    # to sum to one, and then accumulate them in order
    flatten_map = cropped_map.flatten()
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
