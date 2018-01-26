# -*- coding: utf-8 -*-
"""
Resize input image as output window.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import scipy.ndimage
import tensorflow as tf

from niftynet.engine.image_window import ImageWindow, N_SPATIAL
from niftynet.engine.image_window_buffer import InputBatchQueueRunner
from niftynet.layer.base_layer import Layer


class ResizeSampler(Layer, InputBatchQueueRunner):
    """
    This class generates samples by rescaling
    the whole image to the desired size
    currently 5D input is supported:
    ``Height x Width x Depth x time x Modality``
    """

    def __init__(self,
                 reader,
                 data_param,
                 batch_size,
                 spatial_window_size=(),
                 windows_per_image=1,
                 shuffle_buffer=True,
                 queue_length=10):

        self.reader = reader
        self.windows_per_image = windows_per_image
        self.shuffle = bool(shuffle_buffer)

        Layer.__init__(self, name='input_buffer')
        InputBatchQueueRunner.__init__(
            self,
            capacity=queue_length,
            shuffle=self.shuffle)
        tf.logging.info('reading size of preprocessed images')
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
        tf.logging.info('initialised window instance')
        self._create_queue_and_ops(self.window,
                                   enqueue_size=1,
                                   dequeue_size=batch_size)
        tf.logging.info("initialised sampler output %s ", self.window.shapes)

    def layer_op(self, *args, **kwargs):
        """
        This function generates sampling windows to the input buffer
        image data are from ``self.reader()``.

        It first completes window shapes based on image data,
        then resize each image as window and output
        a dictionary (required by input buffer)

        :return: output data dictionary ``{placeholders: data_array}``
        """
        while True:
            image_id, data, interp_orders = self.reader(shuffle=self.shuffle)
            if not data:
                break
            image_shapes = \
                dict((name, data[name].shape) for name in self.window.names)
            # window shapes can be dynamic, here they
            # are converted to static ones
            # as now we know the image shapes
            static_window_shapes = self.window.match_image_shapes(image_shapes)

            # for resize sampler the coordinates are not used
            # simply use the spatial dims of the input image
            all_coordinates = dummy_coordinates(image_id,
                                                static_window_shapes)
            for _ in range(self.windows_per_image):
                output_dict = {}
                for name in list(data):
                    # prepare output dictionary keys
                    coordinates_key = self.window.coordinates_placeholder(name)
                    image_data_key = self.window.image_data_placeholder(name)

                    # prepare coordinates data
                    output_dict[coordinates_key] = all_coordinates[name]

                    # prepare image data
                    image_shape = image_shapes[name]
                    window_shape = static_window_shapes[name]

                    if image_shape == window_shape:
                        # already in the same shape
                        image_window = data[name]
                    else:
                        zoom_ratio = \
                            [float(p) / float(d) for p, d in
                             zip(window_shape, image_shape)]
                        image_window = zoom_3d(
                            image=data[name],
                            ratio=zoom_ratio,
                            interp_order=interp_orders[name][0])
                    output_dict[image_data_key] = image_window[np.newaxis, ...]
                # the output image shape should be
                # [enqueue_batch_size, x, y, z, time, modality]
                # here enqueue_batch_size = 1 as we only have one sample
                # per image
                yield output_dict


def zoom_3d(image, ratio, interp_order):
    """
    Taking 5D image as input, and zoom each 3D slice independently
    """
    assert image.ndim == 5, "input images should be 5D array"
    output = []
    for time_pt in range(image.shape[3]):
        output_mod = []
        for mod in range(image.shape[4]):
            zoomed = scipy.ndimage.zoom(
                image[..., time_pt, mod], ratio[:3], order=interp_order)
            output_mod.append(zoomed[..., np.newaxis, np.newaxis])
        output.append(np.concatenate(output_mod, axis=-1))
    return np.concatenate(output, axis=-2)


def dummy_coordinates(image_id, image_sizes):
    """
    This function returns a set of image window coordinates
    which are just from 0 to image_shapes.
    """
    all_coordinates = {}
    for mod in list(image_sizes):
        starting_coordinates = [0, 0, 0]
        image_spatial_shape = list(image_sizes[mod][:N_SPATIAL])
        coords = [[image_id] + starting_coordinates + image_spatial_shape]
        all_coordinates[mod] = np.asarray(coords)
    return all_coordinates
