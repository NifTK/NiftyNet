# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.input_buffer import InputBatchQueueRunner
from niftynet.io.image_window import ImageWindow, N_SPATIAL
from niftynet.layer.base_layer import Layer


class GridSampler(Layer, InputBatchQueueRunner):
    """
    This class generators samples by uniformly sampling each input volume
    currently 4D input is supported, Height x Width x Depth x Modality
    """

    def __init__(self,
                 reader,
                 data_param,
                 batch_size,
                 spatial_window_size=(),
                 window_border=()):
        # TODO: padding
        self.reader = reader
        Layer.__init__(self, name='input_buffer')
        InputBatchQueueRunner.__init__(
            self, capacity=batch_size * 2, shuffle=False)
        tf.logging.info('reading size of preprocessed inputs')
        self.window = ImageWindow.from_user_spec(self.reader.input_sources,
                                                 self.reader.shapes,
                                                 self.reader.tf_dtypes,
                                                 data_param)
        if spatial_window_size:
            # override all spatial window defined in input modalities sections
            # this is useful when do inference with a spatial window
            # which is different from the training specifications
            self.window.set_spatial_shape(spatial_window_size)

        self.border_size = complete_spatial_border_size(window_border)
        tf.logging.info('initialised window instance')
        self._create_queue_and_ops(self.window,
                                   enqueue_size=1,
                                   dequeue_size=batch_size)
        tf.logging.info("initialised sampler output {}".format(
            self.window.shapes))

        ## running test
        sess = tf.Session()
        _iter = 0
        for x in self():
            sess.run(self._enqueue_op, feed_dict=x)
            _iter += 1
            print('enqueue {}'.format(_iter))
            if _iter == 2:
                break
        out = sess.run(self.pop_batch_op(batch_size=3))
        print('dequeue')
        print(out['image'].shape)
        print(out['image_location'])
        import pdb;
        pdb.set_trace()

    def layer_op(self):
        while True:
            image_id, data, _ = self.reader(idx=None, shuffle=False)
            if not data:
                break
            image_shapes = {mod: data[mod].shape for mod in self.window.fields}
            static_window_shapes = self.window.match_image_shapes(image_shapes)
            coordinates = grid_spatial_coordinates(
                image_id, image_shapes, static_window_shapes, self.border_size)
            n_locations = coordinates.values()[0].shape[0]
            for i in range(n_locations):
                #  initialise output dict
                output_dict = {}
                for name in list(data):
                    assert coordinates[name].shape[0] == n_locations, \
                        "different number of grid samples from the input" \
                        "images, don't know how to combine them in the queue"
                    x_, y_, z_, _x, _y, _z = coordinates[name][i, 1:]
                    try:
                        image_window = data[name][x_:_x, y_:_y, z_:_z, ...]
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
                    output_dict[coordinates_key] = coordinates[name][[i], ...]
                    output_dict[image_data_key] = image_window[np.newaxis, ...]
                yield output_dict


def grid_spatial_coordinates(subject_id, img_sizes, win_sizes, border_size):
    """
    This function generates all coordinates of feasible windows, with
    step sizes specified in grid_size parameter
    :param subject_id: integer value indicates the position of of this
    image in image_reader.file_list
    :param img_sizes: a dictionary of image shapes, {input_name: shape}
    :param win_sizes: a dictionary of window shapes, {input_name: shape}
    :param border_size:
    :return:
    """
    all_coordinates = {}
    for name, image_shape in img_sizes.items():
        window_shape = win_sizes[name]
        grid_size = [win_size - 2 * border_size for (win_size, border_size)
                     in zip(window_shape, border_size)]
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
        subject_list = np.ones((n_locations, 1), dtype=np.int32) * subject_id
        spatial_coords = np.append(subject_list, spatial_coords, axis=1)
        all_coordinates[name] = spatial_coords
    return all_coordinates


def _enumerate_step_points(starting, ending, win_size, step_size):
    """
    generate all possible sampling size in between starting and ending
    :param starting: integer of starting value
    :param ending: integer of ending value
    :param win_size: integer of window length
    :param step_size: integer of distance between two sampling points
    :return: a set of unique sampling points
    """
    starting = max(int(starting), 0)
    ending = max(int(ending), 0)
    win_size = max(int(win_size), 1)
    step_size = max(int(step_size), 1)
    if starting > ending:
        starting, ending = ending, starting
    sampling_point_set = []
    while (starting + win_size) <= ending:
        sampling_point_set.append(starting)
        starting = starting + step_size
    sampling_point_set.append(np.max((ending - win_size, 0)))
    return np.unique(sampling_point_set).flatten()


def complete_spatial_border_size(input_border_size):
    try:
        input_border_size = map(int, input_border_size)
        while len(input_border_size) < N_SPATIAL:
            input_border_size = input_border_size + (1,)
        input_border_size = tuple(input_border_size[:N_SPATIAL])
    except ValueError:
        tf.logging.fatal("wrong window border size format")
        raise
    return input_border_size