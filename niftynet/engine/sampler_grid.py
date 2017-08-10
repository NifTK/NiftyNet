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

    def __init__(self, reader, data_param, batch_size, spatial_window_size=()):
        # TODO: padding
        self.reader = reader
        Layer.__init__(self, name='input_buffer')
        InputBatchQueueRunner.__init__(self,
                                       capacity=batch_size * 2,
                                       shuffle=True)
        tf.logging.info('reading size of preprocessed inputs')
        self.window = ImageWindow.from_user_spec(self.reader.input_sources,
                                                 self.reader.shapes,
                                                 self.reader.tf_dtypes,
                                                 data_param)
        if spatial_window_size:
            self.window.set_spatial_shape(spatial_window_size)
        import pdb; pdb.set_trace()
        tf.logging.info('initialised window instance')
        self._create_queue_and_ops(self.window, batch_size * 2)
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
            image_id, data = self.reader()
            if not data:
                break
            image_sizes = {
                name: data[name].shape for name in self.window.fields}
            import pdb; pdb.set_trace()
            coordinates = rand_spatial_coordinates(
                image_id, image_sizes,
                self.window.shapes, self.window.n_samples)
            #  initialise output dict
            output_dict = self.window.data_dict()
            # fill output dict with data
            for name in list(data):
                # fill output coordinates
                location_array = output_dict[
                    self.window.coordinates_placeholder(name)]
                location_array[...] = coordinates[name]
                # fill output window array
                image_array = output_dict[
                    self.window.image_data_placeholder(name)]
                for (i, location) in enumerate(location_array[:, 1:]):
                    x_, y_, z_, _x, _y, _z = location
                    try:
                        image_array[i, ...] = \
                            data[name][x_:_x, y_:_y, z_:_z, ...]
                    except ValueError:
                        tf.logging.fatal(
                            "dimensionality miss match in input volumes, "
                            "please specify spatial_window_size with a "
                            "3D tuple and make sure each element is "
                            "smaller than the image length in each dim.")
                        raise
            yield output_dict


def rand_spatial_coordinates(subject_id, img_sizes, win_sizes, n_samples):
    uniq_spatial_size = set([img_size[:N_SPATIAL]
                             for img_size in list(img_sizes.values())])
    if len(uniq_spatial_size) > 1:
        tf.logging.fatal("Don't know how to generate sampling "
                         "locations: Spatial dimensions of the "
                         "grouped input sources are not "
                         "consistent. {}".format(uniq_spatial_size))
        raise NotImplementedError
    uniq_spatial_size = uniq_spatial_size.pop()

    # find spatial window location based on the largest spatial window
    spatial_win_sizes = [win_size[:N_SPATIAL]
                         for win_size in win_sizes.values()]
    spatial_win_sizes = np.asarray(spatial_win_sizes, dtype=np.int32)
    max_spatial_win = np.max(spatial_win_sizes, axis=0)
    max_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    for i in range(0, N_SPATIAL):
        max_coords[:, i] = np.random.randint(
            0, max(uniq_spatial_size[i] - max_spatial_win[i], 1), n_samples)

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
    sampling_point_set = []
    while (starting + win_size) <= ending:
        sampling_point_set.append(starting)
        starting = starting + step_size
    sampling_point_set.append(np.max((ending - win_size, 0)))
    return np.unique(sampling_point_set).flatten()
