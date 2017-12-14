# -*- coding: utf-8 -*-
"""
Sampling image by a sliding window.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import ImageWindow, N_SPATIAL
from niftynet.engine.image_window_buffer import InputBatchQueueRunner
from niftynet.layer.base_layer import Layer


# pylint: disable=too-many-locals
class GridSampler(Layer, InputBatchQueueRunner):
    """
    This class generators ND image samples with a sliding window.
    """

    def __init__(self,
                 reader,
                 data_param,
                 batch_size,
                 spatial_window_size=(),
                 window_border=(),
                 queue_length=10):
        self.batch_size = batch_size
        self.reader = reader
        Layer.__init__(self, name='input_buffer')
        InputBatchQueueRunner.__init__(
            self,
            capacity=queue_length,
            shuffle=False)
        tf.logging.info('reading size of preprocessed inputs')
        self.window = ImageWindow.from_data_reader_properties(
            self.reader.input_sources,
            self.reader.shapes,
            self.reader.tf_dtypes,
            data_param)

        if spatial_window_size:
            # override all spatial window defined in input
            # modalities sections
            # this is useful when do inference with a spatial window
            # which is different from the training specifications
            self.window.set_spatial_shape(spatial_window_size)
        self.border_size = window_border
        tf.logging.info('initialised window instance')
        self._create_queue_and_ops(self.window,
                                   enqueue_size=1,
                                   dequeue_size=batch_size)
        tf.logging.info("initialised sampler output %s", self.window.shapes)

    def layer_op(self):
        while True:
            image_id, data, _ = self.reader(idx=None, shuffle=False)
            if not data:
                break
            image_shapes = {name: data[name].shape
                            for name in self.window.names}
            static_window_shapes = self.window.match_image_shapes(image_shapes)
            coordinates = grid_spatial_coordinates(
                image_id, image_shapes, static_window_shapes, self.border_size)

            # extend the number of sampling locations to be divisible
            # by batch size
            n_locations = list(coordinates.values())[0].shape[0]
            extra_locations = 0
            if (n_locations % self.batch_size) > 0:
                extra_locations = \
                    self.batch_size - n_locations % self.batch_size
            total_locations = n_locations + extra_locations

            tf.logging.info(
                'grid sampling image sizes: %s', image_shapes)
            tf.logging.info(
                'grid sampling window sizes: %s', static_window_shapes)
            if extra_locations > 0:
                tf.logging.info(
                    "yielding %d locations from image, "
                    "extended to %d to be divisible by batch size %d",
                    n_locations, total_locations, self.batch_size)
            else:
                tf.logging.info(
                    "yielding %s locations from image", n_locations)

            for i in range(total_locations):
                idx = i % n_locations
                #  initialise output dict
                output_dict = {}
                for name in list(data):
                    assert coordinates[name].shape[0] == n_locations, \
                        "different number of grid samples from the input" \
                        "images, don't know how to combine them in the queue"
                    x_start, y_start, z_start, x_end, y_end, z_end = \
                        coordinates[name][idx, 1:]
                    try:
                        image_window = data[name][
                            x_start:x_end, y_start:y_end, z_start:z_end, ...]
                    except ValueError:
                        tf.logging.fatal(
                            "dimensionality miss match in input volumes, "
                            "please specify spatial_window_size with a "
                            "3D tuple and make sure each element is "
                            "smaller than the image length in each dim.")
                        raise
                    # fill output dict with data
                    coordinates_key = self.window.coordinates_placeholder(name)
                    image_data_key = self.window.image_data_placeholder(name)
                    output_dict[coordinates_key] = coordinates[name][[idx], ...]
                    output_dict[image_data_key] = image_window[np.newaxis, ...]
                yield output_dict


def grid_spatial_coordinates(subject_id, img_sizes, win_sizes, border_size):
    """
    This function generates all coordinates of feasible windows, with
    step sizes specified in grid_size parameter.

    The border size changes the sampling locations but not the
    corresponding window sizes of the coordinates.

    :param subject_id: integer value indicates the position of of this
        image in ``image_reader.file_list``
    :param img_sizes: a dictionary of image shapes, ``{input_name: shape}``
    :param win_sizes: a dictionary of window shapes, ``{input_name: shape}``
    :param border_size: size of padding on both sides of each dim
    :return:
    """
    all_coordinates = {}
    for name, image_shape in img_sizes.items():
        window_shape = win_sizes[name]
        grid_size = [max(win_size - 2 * border, 0)
                     for (win_size, border) in zip(window_shape, border_size)]
        assert len(image_shape) >= N_SPATIAL, 'incompatible image shapes'
        assert len(window_shape) >= N_SPATIAL, 'incompatible window shapes'
        assert len(grid_size) >= N_SPATIAL, 'incompatible step sizes'
        steps_along_each_dim = [
            _enumerate_step_points(starting=0,
                                   ending=image_shape[i],
                                   win_size=window_shape[i],
                                   step_size=grid_size[i])
            for i in range(N_SPATIAL)]
        starting_coords = np.asanyarray(np.meshgrid(*steps_along_each_dim))
        starting_coords = starting_coords.reshape((N_SPATIAL, -1)).T
        n_locations = starting_coords.shape[0]
        # prepare the output coordinates matrix
        spatial_coords = np.zeros((n_locations, N_SPATIAL * 2), dtype=np.int32)
        spatial_coords[:, :N_SPATIAL] = starting_coords
        for idx in range(N_SPATIAL):
            spatial_coords[:, N_SPATIAL + idx] = \
                starting_coords[:, idx] + window_shape[idx]
        max_coordinates = np.max(spatial_coords, axis=0)[N_SPATIAL:]
        assert np.all(max_coordinates <= image_shape[:N_SPATIAL]), \
            "window size greater than the spatial coordinates {} : {}".format(
                max_coordinates, image_shape)
        subject_list = np.ones((n_locations, 1), dtype=np.int32) * subject_id
        spatial_coords = np.append(subject_list, spatial_coords, axis=1)
        all_coordinates[name] = spatial_coords
    return all_coordinates


def _enumerate_step_points(starting, ending, win_size, step_size):
    """
    generate all possible sampling size in between starting and ending.

    :param starting: integer of starting value
    :param ending: integer of ending value
    :param win_size: integer of window length
    :param step_size: integer of distance between two sampling points
    :return: a set of unique sampling points
    """
    try:
        starting = max(int(starting), 0)
        ending = max(int(ending), 0)
        win_size = max(int(win_size), 1)
        step_size = max(int(step_size), 1)
    except (TypeError, ValueError):
        tf.logging.fatal(
            'step points should be specified by integers, received:'
            '%s, %s, %s, %s', starting, ending, win_size, step_size)
        raise ValueError
    if starting > ending:
        starting, ending = ending, starting
    sampling_point_set = []
    while (starting + win_size) <= ending:
        sampling_point_set.append(starting)
        starting = starting + step_size
    additional_last_point = ending - win_size
    sampling_point_set.append(max(additional_last_point, 0))
    sampling_point_set = np.unique(sampling_point_set).flatten()
    if len(sampling_point_set) == 2:
        # in case of too few samples, adding
        # an additional sampling point to
        # the middle between starting and ending
        sampling_point_set = np.append(
            sampling_point_set, np.round(np.mean(sampling_point_set)))
    _, uniq_idx = np.unique(sampling_point_set, return_index=True)
    return sampling_point_set[np.sort(uniq_idx)]
