# -*- coding: utf-8 -*-
"""
Generating uniformly distributed image window from input image
This can also be considered as a "random cropping" layer of the
input image.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import ImageWindow, N_SPATIAL
from niftynet.engine.image_window_buffer import InputBatchQueueRunner
from niftynet.layer.base_layer import Layer


# pylint: disable=too-many-arguments
class UniformSampler(Layer, InputBatchQueueRunner):
    """
    This class generates samples by uniformly sampling each input volume
    currently the coordinates are randomised for spatial dims only,
    i.e., the first three dims of image.

    This layer can be considered as a "random cropping" layer of the
    input image.
    """

    def __init__(self,
                 reader,
                 data_param,
                 batch_size,
                 windows_per_image,
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

        tf.logging.info('initialised window instance')
        self._create_queue_and_ops(self.window,
                                   enqueue_size=windows_per_image,
                                   dequeue_size=batch_size)
        tf.logging.info("initialised sampler output %s ", self.window.shapes)

        self.spatial_coordinates_generator = rand_spatial_coordinates

    # pylint: disable=too-many-locals
    def layer_op(self):
        """
        This function generates sampling windows to the input buffer
        image data are from ``self.reader()``

        It first completes window shapes based on image data,
        then finds random coordinates based on the window shapes
        finally extract window with the coordinates and output
        a dictionary (required by input buffer).

        :return: output data dictionary ``{placeholders: data_array}``
        """
        while True:
            image_id, data, _ = self.reader(idx=None, shuffle=True)
            if not data:
                break
            image_shapes = dict((name, data[name].shape)
                                for name in self.window.names)
            static_window_shapes = self.window.match_image_shapes(image_shapes)

            # find random coordinates based on window and image shapes
            coordinates = self.spatial_coordinates_generator(
                image_id,
                data,
                image_shapes,
                static_window_shapes,
                self.window.n_samples)

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
                if len(image_array) > 1:
                    output_dict[image_data_key] = \
                        np.concatenate(image_array, axis=0)
                else:
                    output_dict[image_data_key] = image_array[0]
            # the output image shape should be
            # [enqueue_batch_size, x, y, z, time, modality]
            # where enqueue_batch_size = windows_per_image
            yield output_dict


def rand_spatial_coordinates(subject_id,
                             data,
                             img_sizes,
                             win_sizes,
                             n_samples=1):
    """
    ``win_sizes`` could be different (for example in segmentation network
    input image window size is ``32x32x10``,
    training label window is ``16x16x10`` -- the network reduces x-y plane
    spatial resolution.)

    This function handles this situation by first find the largest
    window across these window definitions, and generate the coordinates.
    These coordinates are then adjusted for each of the
    smaller window sizes (the output windows are concentric).
    """
    assert data is not None, "No input from image reader. Please check" \
                             "the configuration file."
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
    max_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    for i in range(0, N_SPATIAL):
        assert uniq_spatial_size[i] >= max_spatial_win[i], \
            "window size {} is larger than image size {}".format(
                max_spatial_win[i], uniq_spatial_size[i])
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
        spatial_coords = np.append(subject_id[:, None], spatial_coords, axis=1)
        all_coordinates[mod] = spatial_coords
    return all_coordinates
