# -*- coding: utf-8 -*-
"""
Generating sample arrays from random distributions
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import ImageWindow, N_SPATIAL
from niftynet.engine.image_window_buffer import InputBatchQueueRunner
from niftynet.layer.base_layer import Layer
from scipy import ndimage
from scipy.signal import fftconvolve


# pylint: disable=too-many-arguments
class SelectiveSampler(Layer, InputBatchQueueRunner):
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
                 queue_length=10):
        self.reader = reader
        Layer.__init__(self, name='input_buffer')
        InputBatchQueueRunner.__init__(
            self,
            capacity=queue_length,
            shuffle=True)
        tf.logging.info('reading size of preprocessed images')
        self.window = ImageWindow.from_data_reader_properties(
            self.reader.input_sources,
            self.reader.shapes,
            self.reader.tf_dtypes,
            data_param)
        self.constraint = constraint

        tf.logging.info('initialised window instance')
        self._create_queue_and_ops(self.window,
                                   enqueue_size=windows_per_image,
                                   dequeue_size=batch_size)
        tf.logging.info("initialised sampler output %s "
                        " [-1 for dynamic size]", self.window.shapes)

    # pylint: disable=too-many-locals
    def layer_op(self):
        """
        This function generates sampling windows to the input buffer
        image data are from self.reader()
        it first completes window shapes based on image data,
        then finds random coordinates based on the window shapes
        finally extract window with the coordinates and output
        a dictionary (required by input buffer)
        :return: output data dictionary {placeholders: data_array}
        """
        while True:
            image_id, data, _ = self.reader(idx=None, shuffle=True)
            if not data:
                break
            image_shapes = {
                name: data[name].shape for name in self.window.names}
            print(image_shapes)
            static_window_shapes = self.window.match_image_shapes(image_shapes)
            candidates, proba_cand = candidate_indices(static_window_shapes[
                                                       'label'],
                                                       data['label'],
                                                       self.constraint)
            if not np.sum(candidates) >= self.window.n_samples:
                print('Constraints not fulfilled for this case')
                continue

            # find random coordinates based on window, potential candidates and
            # image shapes
            coordinates = rand_choice_coordinates(
                image_id, image_shapes,
                static_window_shapes, candidates, self.window.n_samples,
                mean_counts_size=proba_cand)

            # initialise output dict, placeholders as dictionary keys
            # this dictionary will be used in
            # enqueue operation in the form of: `feed_dict=output_dict`
            output_dict = {}
            # fill output dict with data
            for name in list(data):
                coordinates_key = self.window.coordinates_placeholder(name)
                image_data_key = self.window.image_data_placeholder(name)

                # fill the coordinates
                location_array = coordinates[name]
                output_dict[coordinates_key] = location_array

                # fill output window array
                image_array = []
                for window_id in range(self.window.n_samples):
                    x_start, y_start, z_start, x_end, y_end, z_end = \
                        location_array[window_id, 1:]
                    try:
                        image_window = data[name][
                            x_start:x_end, y_start:y_end, z_start:z_end, ...]
                        image_array.append(image_window[np.newaxis, ...])
                    except ValueError:
                        tf.logging.fatal(
                            "dimensionality miss match in input volumes, "
                            "please specify spatial_window_size with a "
                            "3D tuple and make sure each element is "
                            "smaller than the image length in each dim.")
                        raise
                    if name == 'label':
                        print('Check of constraint validity')
                        image_window = data[name][
                                       x_start:x_end, y_start:y_end,
                                       z_start:z_end, ...]
                        print(check_constraint(image_window, self.constraint))
                        # image_window = data[name][
                        #                x_start+h:x_end+h, y_start+h:y_end+h,
                        #                z_start:z_end, ...]
                        # print(check_constraint(image_window, self.constraint))
                if len(image_array) > 1:
                    output_dict[image_data_key] = \
                        np.concatenate(image_array, axis=0)
                else:
                    output_dict[image_data_key] = image_array[0]
            # the output image shape should be
            # [enqueue_batch_size, x, y, z, time, modality]
            # where enqueue_batch_size = windows_per_image
            yield output_dict


def create_label_size_map(data, value):
    '''
    This function creates the maps of label size. For each connected
    component of a label with value :value:, the binary segmentation is
    replaced by the size of the considered element
    :param data: segmentation
    :param value: value of the label to consider
    :return: count_data
    '''
    data = np.round(data)
    labels = np.unique(data)
    print("Labels are ", labels)
    binary_seg = np.copy(data)
    binary_seg = np.where(binary_seg == value, np.ones_like(data),
                          np.zeros_like(data))
    labelled_data, num_features = ndimage.label(binary_seg)
    print("Labelling features done")
    unique, count = np.unique(labelled_data, return_counts=True)
    count_data = np.copy(labelled_data)
    for u, c in zip(unique, count):
        if u != 0:
            count_data = np.where(labelled_data == u, np.ones_like(data)*c,
                                  count_data)
    return count_data


def check_constraint(data, constraint):
    unique, count = np.unique(np.round(data), return_counts=True)
    list_labels = []
    data = np.round(data)
    if constraint.list_labels is not None:
        list_labels = constraint.list_labels
        for label in constraint.list_labels:
            if label not in unique:
                print('Label %d is not there' % label)
                return False
    num_labels_add = 0
    if constraint.num_labels > 0:
        num_labels_add = constraint.num_labels - len(list_labels)
        if num_labels_add <= 0:
            num_labels_add = 0
        if len(unique) < constraint.num_labels:
            print('Missing labels')
            return False
    to_add = num_labels_add
    if constraint.min_ratio > 0:
        num_min = constraint.min_number_from_ratio(data.shape)
        print('unique in test is ', unique)
        for value, c in zip(unique, count):
            if value in list_labels:
                if c < num_min:
                    print('Not enough in label %d', value)
                    return False
            else:
                if c > num_min:
                    to_add -= 1
        if to_add > 0:
            print('to add initial is ', num_labels_add)
            print('Not enough in additional labels')
            return False
    return True


def candidate_indices(win_sizes, data, constraint):
    '''
    This functions creates a binary map of potential candidate indices given
    the specified constraints and the recalculated probability to select each of
     these candidates so as to uniformise the sampling according to the size of
     connected elements
    :param win_sizes:
    :param data: segmentation
    :param constraint: sampling constraint
    :return: candidates: binary map of potential indices, proba_fin:
    corresponding maps of associated sampling probability
    '''
    unique = np.unique(np.round(data))
    list_labels = []
    data = np.round(data)
    if constraint.list_labels is not None:
        list_labels = constraint.list_labels
        for label in constraint.list_labels:
            if label not in unique:
                print('Label %d is not there' % label)
                return np.zeros_like(data)
    num_labels_add = 0
    if constraint.num_labels > 0:
        num_labels_add = constraint.num_labels - len(list_labels)
        if num_labels_add <= 0:
            num_labels_add = 0
        if len(unique) < constraint.num_labels:
            print('Missing labels')
            return np.zeros_like(data)
    if constraint.min_ratio > 0:
        num_min = constraint.min_number_from_ratio(win_sizes)
        spatial_win_sizes = win_sizes[:N_SPATIAL]
        # spatial_win_sizes = [win_size[:N_SPATIAL]
        #                      for win_size in win_sizes.values()]
        spatial_win_sizes = np.asarray(spatial_win_sizes, dtype=np.int32)
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
        # print(shape_ones, padding)
        final = np.pad(np.ones(shape_ones), np.asarray(padding,
                                                       dtype=np.int32),
                       'constant')
        new_win_size = np.copy(win_sizes)
        # new_win_size[:N_SPATIAL] = win_sizes[0]/8
        window_mean = np.ones(new_win_size, dtype=np.int32)
        mean_counts_size = []
        # print(unique)
        for value in unique:
            # print(np.sum(data), 'sum in data', np.prod(data.shape),
            #       ' elements in data')
            seg_label = np.copy(data)
            seg_label = np.asarray(seg_label, dtype=np.int32)
            # print(np.sum(seg_label))
            seg_label = np.where(seg_label == value, np.ones_like(data),
                           np.zeros_like(
                data))
            print(np.sum(seg_label), " num values in seg_label ", value)
            label_size = create_label_size_map(seg_label, 1)
            # print(value, np.sum(seg_label), seg_label.shape,
            #       window_mean.shape, num_min)
            print('Begin fft convolve')
            counts_window = fftconvolve(seg_label, window_mean, 'same')
            print('finished fft convolve')
            valid_places = np.where(counts_window > np.max([num_min, 1]),
                                    np.ones_like(data), np.zeros_like(data))
            counts_size = fftconvolve(label_size * valid_places, window_mean,
                                      'same')
            mean_counts_size_temp = np.nan_to_num(
                counts_size * 1.0 / counts_window)
            mean_counts_size_temp = np.where(counts_window == 0, np.zeros_like(
                data), mean_counts_size_temp)
            # print(np.max(counts_size), " max size")
            # print(np.sum(valid_places), value)
            if value in list_labels:
                # print(value, 'in list_labels')
                mean_counts_size.append(mean_counts_size_temp)
                final = valid_places * final
                print('final calculated for value in list_labels')
            else:
                list_counts.append(valid_places)
        # print(len(list_counts))
        print('final characterisation')
        for i in range(0, len(list_counts)):
            # print(final.shape, list_counts[i].shape, np.max(final), np.max(
            #     list_counts[i]))
            final += list_counts[i]
        print('initialising candidates', num_labels_add)
        candidates = np.zeros_like(data, dtype=np.int32)
        candidates[final >= num_labels_add+1] = 1
        print(np.sum(candidates), 'number of candidates')
        proba_fin = create_probability_weights(candidates, mean_counts_size)
        return candidates, proba_fin


def create_probability_weights(candidates, mean_counts_size):
    '''
    This functions creates the probability weighting given the valid
    candidates and the size of connected components associated to this candidate
    :param candidates: binary map of the valid candidates
    :param mean_counts_size: counts attributed to each candidate
    :return: probability map for the selection of any voxel as candidate
    '''
    proba_weight = np.ones_like(candidates)
    for i in range(0, len(mean_counts_size)):
        print(candidates.shape, mean_counts_size[i].shape, np.max(candidates),
              np.max(mean_counts_size[i]))

        possible_mean_count = np.nan_to_num(candidates*mean_counts_size[i])
        max_count = np.ceil(np.max(possible_mean_count))
        # print(max_count , 'is max_count')
        unique, counts = np.unique(np.round(possible_mean_count),
                                   return_counts=True)
        numb_counts = np.sum(counts[1:])
        reciprocal_hist = numb_counts * np.reciprocal(np.asarray(counts[1:],
                                                                 dtype=
                                                                 np.float32))
        sum_rec = np.sum(reciprocal_hist)
        proba_hist = np.divide(reciprocal_hist, sum_rec)
        print(unique, counts, sum_rec, len(proba_hist))
        e_start = unique[1:]
        e_end = unique[2:]
        e_end.tolist().append(max_count+1)
        print(e_start.shape, e_end.shape, e_start[-1], e_end[-1], len(
            proba_hist))
        candidates_proba = np.zeros_like(candidates)
        for (e_s, e_e, h) in zip(e_start, e_end, proba_hist):
            candidates_proba = np.where((possible_mean_count >= e_s) *
                                        (possible_mean_count < e_e),
                                        h * np.ones_like(candidates),
                                        candidates_proba)
        proba_weight = np.multiply(proba_weight, candidates_proba)
        print("Finished probability calculation")
    return np.divide(proba_weight, np.sum(proba_weight))


def rand_choice_coordinates(subject_id, img_sizes, win_sizes,
                            candidates, n_samples=1, mean_counts_size=None):
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
    # n_samples = 1
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
    print('max_spatial_win is ', max_spatial_win)
    max_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    candidates_indices = np.vstack(np.where(candidates == 1)).T
    list_indices = np.arange(len(candidates_indices))
    # print(np.sum(candidates), list_indices)
    # print(len(candidates_indices), candidates_indices.shape)
    proba = []
    for (c,  p) in zip(candidates.flatten(), mean_counts_size.flatten()):
        if c >= 1:
            proba.append(p)

    if mean_counts_size is not None:
        list_indices_fin = np.random.choice(list_indices, n_samples,
                                            replace=False, p=proba)
    else:
        list_indices_fin = list_indices
        np.random.shuffle(list_indices)
    for i in range(0, n_samples):
        indices_to_add = candidates_indices[list_indices_fin[i]]
        print(max_coords.shape, indices_to_add)
        for s in range(0, N_SPATIAL):
            max_coords[i, s] = indices_to_add[s] - np.floor(
                spatial_win_sizes[0]/2)[s]
    # for i in range(0, N_SPATIAL):
    #     assert uniq_spatial_size[i] >= max_spatial_win[i], \
    #         "window size {} is larger than image size {}".format(
    #             max_spatial_win[i], uniq_spatial_size[i])
    #     max_coords[:, i] = np.random.randint(
    #         0, max(uniq_spatial_size[i] - max_spatial_win[i], 1), n_samples)

    # adjust max spatial coordinates based on each spatial window size
    all_coordinates = {}
    for mod in list(win_sizes):
        win_size = win_sizes[mod][:N_SPATIAL]
        half_win_diff = np.floor((max_spatial_win - win_size) / 2.0)
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
    print('Finished sampling')
    return all_coordinates


class Constraint():
    def __init__(self,
                 compulsory_labels=[0, 1],
                 min_ratio=0.000001,
                 min_num_labels=2):
        self.list_labels = compulsory_labels
        self.min_ratio = min_ratio
        self.num_labels = min_num_labels

    def min_number_from_ratio(self, win_size):
        num_elements = np.prod(win_size)
        print(num_elements, self.min_ratio)
        min_num = np.ceil(self.min_ratio * num_elements)
        return min_num

    def num_labels_to_add(self):
        labels_to_add = self.num_labels - len(self.list_labels)
        return np.max([0, labels_to_add])

