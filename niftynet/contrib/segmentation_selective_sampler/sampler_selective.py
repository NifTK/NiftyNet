# -*- coding: utf-8 -*-
"""
Generating sample arrays from random distributions
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf
from scipy import ndimage
from scipy.signal import fftconvolve

from niftynet.engine.image_window import N_SPATIAL
from niftynet.engine.sampler_uniform_v2 import UniformSampler


# pylint: disable=too-many-arguments
class SelectiveSampler(UniformSampler):
    """
    This class generators samples by uniformly sampling each input volume
    currently the coordinates are randomised for spatial dims only,
    i.e., the first three dims of image.

    This layer can be considered as a `random cropping` layer of the
    input image.
    """

    def __init__(self,
                 reader,
                 data_param,
                 batch_size,
                 windows_per_image,
                 constraint,
                 random_windows_per_image=0,
                 queue_length=10):
        UniformSampler.__init__(self,
                                reader=reader,
                                data_param=data_param,
                                batch_size=batch_size,
                                windows_per_image=windows_per_image,
                                queue_length=queue_length)
        self.constraint = constraint
        self.n_samples_rand = random_windows_per_image
        self.spatial_coordinates_generator = \
            self.selective_spatial_coordinates()
        tf.logging.info('initialised selective sampling')

    def selective_spatial_coordinates(self):
        """
        this function generates a function, the function will be used
        to replace the random coordinate generated in UniformSampler

        This wrapper is created in order to use self.properties
        directly in the generator.

        :return: a spatial coordinates generator function
        """
        print('self rand samples is ', self.n_samples_rand)

        def spatial_coordinates_function(
                subject_id,
                data,
                img_sizes,
                win_sizes,
                n_samples=1,
                n_samples_rand=0):
            """
            this function first find a set of feasible locations,
            based on discrete labels and randomly choose n_samples
            from the feasible locations.

            :param subject_id:
            :param data:
            :param img_sizes:
            :param win_sizes:
            :param n_samples: total number of samples
            :param n_samples_rand: number of totally random samples apart
            from selected ones
            :return:
            """
            candidates, proba_cand = candidate_indices(
                win_sizes['label'], data['label'], self.constraint)
            print('feasible locations found')
            if np.sum(candidates) < self.window.n_samples:
                print('Constraints not fulfilled for this case')
                # return something here...

            # find random coordinates based on window, potential
            # candidates and image shapes

            coordinates = rand_choice_coordinates(subject_id,
                                                  img_sizes,
                                                  win_sizes,
                                                  candidates,
                                                  proba_cand,
                                                  n_samples,
                                                  n_samples_rand)
            return coordinates

        return spatial_coordinates_function


def create_label_size_map(data):
    """
    This function creates the maps of label size. For each connected
    component of a label with value :value:, the binary segmentation is
    replaced by the size of the considered element
    :param data: segmentation
    :param value: value of the label to consider
    :return: count_data
    """
    labelled_data, _ = ndimage.label(data)
    components, count = np.unique(labelled_data, return_counts=True)
    count_data = np.copy(labelled_data)
    for label, size in zip(components, count):
        if label == 0:
            continue
        count_data[labelled_data == label] = size
    return count_data


def candidate_indices(win_sizes, data, constraint):
    """
    This functions creates a binary map of potential candidate indices given
    the specified constraints and the recalculated probability to select each of
     these candidates so as to uniformise the sampling according to the size of
     connected elements
    :param win_sizes:
    :param data: segmentation
    :param constraint: sampling constraint
    :return: candidates: binary map of potential indices, proba_fin:
    corresponding maps of associated sampling probability
    """
    unique = np.unique(np.round(data))
    list_labels = []
    data = np.round(data)
    if constraint.list_labels:
        list_labels = constraint.list_labels

        print('list labels is ', list_labels)
        for label in list_labels:
            if label not in unique:
                print('Label %d is not there' % label)
                return np.zeros_like(data), None
    num_labels_add = max(constraint.num_labels - len(list_labels), 0) \
        if constraint.num_labels > 0 else 0
    if len(unique) < constraint.num_labels:
        print('Missing labels')
        return np.zeros_like(data), None
    if constraint.min_ratio > 0:
        num_min = constraint.min_number_from_ratio(win_sizes)
        spatial_win_sizes = np.asarray(win_sizes[:N_SPATIAL], dtype=np.int32)
        max_spatial_win = spatial_win_sizes[0]
        # Create segmentation for this label
        list_counts = []
        shape_ones = np.asarray(data.shape)
        # print(shape_ones, max_spatial_win)
        half_max_size = np.floor(max_spatial_win / 2)
        padding = []
        for i in range(0, len(win_sizes)):
            if i < N_SPATIAL:
                shape_ones[i] -= 2 * half_max_size
                padding = padding + [[half_max_size, half_max_size], ]
            else:
                padding = padding + [[0, 0], ]

        final = np.pad(np.ones(shape_ones),
                       np.asarray(padding, dtype=np.int32),
                       'constant')
        # print(shape_ones, padding, data.shape, np.sum(np.ones(data.shape)),
        #       np.sum(final))
        window_ones = np.ones(win_sizes, dtype=np.int32)
        mean_counts_size = []
        # print(unique)
        for value in unique:
            # print(np.sum(data), 'sum in data', np.prod(data.shape),
            #       ' elements in data')
            seg_label = (data == value).astype(data.dtype)
            # print(np.sum(seg_label), " num values in seg_label ", value)
            label_size = create_label_size_map(seg_label)
            # print(value, np.sum(seg_label), seg_label.shape,
            #       window_ones.shape, num_min)
            # print('Begin fft convolve')
            counts_window = fftconvolve(seg_label, window_ones, 'same')
            # print('finished fft convolve')
            valid_places = \
                (counts_window > np.max([num_min, 1])).astype(data.dtype)
            counts_size = fftconvolve(
                label_size * valid_places, window_ones, 'same')
            mean_counts_size_temp = np.nan_to_num(
                counts_size * 1.0 / counts_window)
            mean_counts_size_temp[counts_window == 0] = 0
            # print(np.max(counts_size), " max size")
            # print(np.sum(valid_places), value)
            if value in list_labels:
                # print(value, 'in list_labels')
                mean_counts_size.append(mean_counts_size_temp)
                # print(np.sum(valid_places))
                valid_places = np.where(final == 0, np.zeros_like(final),
                                        valid_places)
                final = np.copy(valid_places)
                # print(np.sum(valid_places))
                print('final calculated for value in list_labels')
            else:
                list_counts.append(valid_places)
        list_counts.append(final)
        # print(len(list_counts))
        print('final characterisation')
        # for i in range(0, len(list_counts)):
        #     # print(final.shape, list_counts[i].shape, np.max(final), np.max(
        #     #     list_counts[i]))
        #     final += list_counts[i]
        final = np.sum(np.asarray(list_counts), axis=0)
        # print(final.shape, 'shape of final', len(list_counts))
        print('initialising candidates', num_labels_add)
        candidates = np.where(final >= num_labels_add + 1, np.ones_like(final),
                              np.zeros_like(final))
        # candidates[final >= num_labels_add + 1] = 1
        print(np.sum(candidates), 'number of candidates')
        proba_fin = None
        if constraint.proba_connected:
            proba_fin = create_probability_weights(candidates, mean_counts_size)
        return candidates, proba_fin


def create_probability_weights(candidates, mean_counts_size):
    """
    This functions creates the probability weighting given the valid
    candidates and the size of connected components associated to this candidate
    :param candidates: binary map of the valid candidates
    :param mean_counts_size: counts attributed to each candidate
    :return: probability map for the selection of any voxel as candidate
    """
    proba_weight = np.ones_like(candidates)
    for i in range(0, len(mean_counts_size)):
        # print(candidates.shape, mean_counts_size[i].shape, np.max(candidates),
        #       np.max(mean_counts_size[i]))

        possible_mean_count = np.nan_to_num(candidates * mean_counts_size[i])
        max_count = np.ceil(np.max(possible_mean_count))
        print(max_count , 'is max_count')
        unique, counts = np.unique(np.round(possible_mean_count),
                                   return_counts=True)
        reciprocal_hist = np.sum(counts[1:]) * np.reciprocal(
            np.asarray(counts[1:], dtype=np.float32))
        sum_rec = np.sum(reciprocal_hist)
        proba_hist = np.divide(reciprocal_hist, sum_rec)
        print(unique, counts, sum_rec, len(proba_hist))
        e_start = unique[1:]
        e_end = unique[2:]
        e_end.tolist().append(max_count + 1)
        # print(e_start.shape, e_end.shape, e_start[-1], e_end[-1], len(
        #     proba_hist))
        candidates_proba = np.zeros_like(candidates)
        if len(unique) == 2:
            proba_weight = np.ones_like(candidates, dtype=np.float32) * \
                           1.0/np.sum(candidates)
            proba_weight = np.multiply(proba_weight, candidates)
        else:
            for (e_s, e_e, size) in zip(e_start, e_end, proba_hist):
                prob_selector = \
                    (possible_mean_count >= e_s) & (possible_mean_count < e_e)
                candidates_proba[prob_selector.astype(np.bool)] = size
            proba_weight = np.multiply(proba_weight, candidates_proba)
        print("Finished probability calculation")
    return np.divide(proba_weight, np.sum(proba_weight))


def rand_choice_coordinates(subject_id,
                            img_sizes,
                            win_sizes,
                            candidates,
                            mean_counts_size=None,
                            n_samples=1, n_samples_rand=0):
    """
    win_sizes could be different, for example in segmentation network
    input image window size is 32x32x10,
    training label window is 16x16x10, the network reduces x-y plane
    spatial resolution.
    This function handles this situation by first find the largest
    window across these window definitions, and generate the coordinates.
    These coordinates are then adjusted for each of the
    smaller window sizes (the output windows are concentric).
    """
    print(n_samples)
    n_samples_rand = np.min([n_samples_rand, n_samples-1])
    n_samples_cand = n_samples - n_samples_rand
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
    for i in range(0, N_SPATIAL):
        assert uniq_spatial_size[i] >= max_spatial_win[i], \
            "window size {} is larger than image size {}".format(
                max_spatial_win[i], uniq_spatial_size[i])

    # randomly choose n_samples from candidate locations
    candidates_indices = np.vstack(np.where(candidates == 1)).T
    # print(np.min(candidates_indices), np.max(candidates_indices),
    # len(candidates_indices))
    list_indices_fin = np.arange(len(candidates_indices))
    if mean_counts_size is not None:
        # Probability weighting considered
        print(np.sum(mean_counts_size), 'proba weighting considered')
        proba = [p for (c, p)
                 in zip(candidates.flatten(), mean_counts_size.flatten())
                 if c >= 1]
        list_indices_fin = np.random.choice(
            list_indices_fin, n_samples_cand, replace=False, p=proba)
    else:
        np.random.shuffle(list_indices_fin)
        list_indices_fin = list_indices_fin[:n_samples_cand]
    if n_samples_rand > 0:
        spatial_win_sizes = np.asarray(win_sizes[:N_SPATIAL], dtype=np.int32)
        max_spatial_win = spatial_win_sizes[0]
        # Create segmentation for this label
        # list_counts = []
        shape_ones = np.asarray(candidates.shape)
        # print(shape_ones, max_spatial_win)
        half_max_size = np.floor(max_spatial_win / 2)
        padding = []
        for i in range(0, len(shape_ones)):
            if i < N_SPATIAL:
                shape_ones[i] -= 2 * half_max_size
                padding = padding + [[half_max_size, half_max_size], ]
            else:
                padding = padding + [[0, 0], ]
        # print(shape_ones, padding)
        rand_one = np.pad(np.ones(shape_ones),
                          np.asarray(padding, dtype=np.int32),
                          'constant')
        # rand_one = np.ones_like(candidates)
        list_possible_rand = np.vstack(np.where(rand_one == 1)).T
        list_indices_rand = np.random.choice(list_possible_rand, n_samples_rand,
                                             replace=False)
        list_indices_fin = np.concatenate((list_indices_fin, list_indices_rand))

    max_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    half_win = np.floor(np.asarray(win_sizes['image']) / 2).astype(np.int)
    for (i_sample, ind) in enumerate(list_indices_fin):
        max_coords[i_sample, :N_SPATIAL] = \
            candidates_indices[ind][:N_SPATIAL] - half_win[:N_SPATIAL]
    # print(np.min(max_coords), np.max(max_coords))
    # adjust max spatial coordinates based on each spatial window size
    all_coordinates = {}
    for mod in list(win_sizes):
        win_size = win_sizes[mod][:N_SPATIAL]
        half_win_diff = np.floor((max_spatial_win - win_size) / 2.0)
        # print(win_size, half_win_diff, 'win and half_Win_diff')
        # shift starting coords of the window
        # so that smaller windows are centred within the large windows
        spatial_coords = np.zeros((n_samples, N_SPATIAL * 2), dtype=np.int32)
        spatial_coords[:, :N_SPATIAL] = \
            max_coords[:, :N_SPATIAL] + half_win_diff[:N_SPATIAL]
        spatial_coords[:, N_SPATIAL:] = \
            spatial_coords[:, :N_SPATIAL] + win_size[:N_SPATIAL]
        # include the subject id
        subject_id = np.ones((n_samples,), dtype=np.int32) * subject_id
        spatial_coords = np.append(
            subject_id[:, None], spatial_coords, axis=1)
        all_coordinates[mod] = spatial_coords
    # print(all_coordinates)
    return all_coordinates


class Constraint(object):
    """
    group of user specified constraints for choosing window samples
    """

    def __init__(self,
                 compulsory_labels=(0, 1),
                 min_ratio=0.000001,
                 min_num_labels=2,
                 flag_proba_uniform=False):
        self.list_labels = compulsory_labels
        self.min_ratio = min_ratio
        self.num_labels = min_num_labels
        self.proba_connected = flag_proba_uniform

    def min_number_from_ratio(self, win_size):
        """
        number of voxels from ratio
        :param win_size:
        :return:
        """
        num_elements = np.prod(win_size)
        print(num_elements, self.min_ratio)
        min_num = np.ceil(self.min_ratio * num_elements)
        return min_num
