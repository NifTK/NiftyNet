# -*- coding: utf-8 -*-
"""
Resize input image as output window.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import scipy.ndimage
import tensorflow as tf

from niftynet.contrib.dataset_sampler.image_window_dataset import \
    ImageWindowDataset
from niftynet.engine.image_window import LOCATION_FORMAT


class ResizeSampler(ImageWindowDataset):
    """
    This class generates samples by rescaling
    the whole image to the desired size
    Assuming the reader's output is 5d:
    ``Height x Width x Depth x time x Modality``
    """

    def __init__(self,
                 reader,
                 window_sizes,
                 batch_size=1,
                 spatial_window_size=None,
                 windows_per_image=1,
                 shuffle=True,
                 queue_length=10,
                 name='resize_sampler_v2'):
        tf.logging.info('reading size of preprocessed images')
        ImageWindowDataset.__init__(
            self,
            reader=reader,
            window_sizes=window_sizes,
            batch_size=batch_size,
            windows_per_image=windows_per_image,
            queue_length=queue_length,
            shuffle=shuffle,
            epoch=-1 if shuffle else 1,
            name=name)
        if spatial_window_size:
            # override all spatial window defined in input
            # modalities sections
            # this is useful when do inference with a spatial window
            # which is different from the training specifications
            self.window.set_spatial_shape(spatial_window_size)
        tf.logging.info("initialised resize sampler %s ", self.window.shapes)

    def layer_op(self, idx=None):
        """
        This function generates sampling windows to the input buffer
        image data are from ``self.reader()``.

        It first completes window shapes based on image data,
        then resize each image as window and output
        a dictionary (required by input buffer)

        :return: output data dictionary ``{'image_modality': data_array}``
        """
        image_id, data, interp_orders = self.reader(idx=idx)
        image_shapes = \
            dict((name, data[name].shape) for name in self.window.names)
        # window shapes can be dynamic, here they
        # are converted to static ones
        # as now we know the image shapes
        static_window_shapes = self.window.match_image_shapes(image_shapes)

        # for resize sampler the coordinates are not used
        # simply use the spatial dims of the input image
        output_dict = {}
        for name in list(data):
            # prepare output dictionary keys
            coordinates_key = LOCATION_FORMAT.format(name)
            image_data_key = name

            output_dict[coordinates_key] = self.dummy_coordinates(
                image_id, static_window_shapes[name], self.window.n_samples)
            image_array = []
            for _ in range(self.window.n_samples):
                # prepare image data
                image_shape = image_shapes[name]
                window_shape = static_window_shapes[name]

                if image_shape == window_shape or interp_orders[name][0] < 0:
                    # already in the same shape
                    image_window = data[name]
                else:
                    zoom_ratio = [float(p) / float(d) for p, d in
                                  zip(window_shape, image_shape)]
                    image_window = self.zoom_3d(
                        image=data[name],
                        ratio=zoom_ratio,
                        interp_order=interp_orders[name][0])
                image_array.append(image_window[np.newaxis, ...])
            if len(image_array) > 1:
                output_dict[image_data_key] = \
                    np.concatenate(image_array, axis=0)
            else:
                output_dict[image_data_key] = image_array[0]
        # the output image shape should be
        # [enqueue_batch_size, x, y, z, time, modality]
        # here enqueue_batch_size = 1 as we only have one sample
        # per image
        return output_dict

    @classmethod
    def zoom_3d(cls, image, ratio, interp_order):
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
