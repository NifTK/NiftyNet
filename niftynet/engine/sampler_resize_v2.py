# -*- coding: utf-8 -*-
"""
Resize input image as output window.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import scipy.ndimage as scnd
import tensorflow as tf

from niftynet.engine.image_window_dataset import ImageWindowDataset
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
                 smaller_final_batch_mode='pad',
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
            smaller_final_batch_mode=smaller_final_batch_mode,
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
                    image_window = zoom_3d(data[name],
                                           zoom_ratio,
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



def zoom_3d(im, ratio, interp_order=1):
    assert (interp_order<4) and (interp_order>-1), "interp_order is between 0 and 3"
    imshape = im.shape
    dim = len(imshape)
    spatial_coord = {}
    size = tuple([int(imshape[i]*ratio[i]) for i in range(len(imshape))])
    for i, s in enumerate(size):
        spatial_coord[i] = imshape[i] / s * (np.arange(0, s) + 0.5) - 0.5
    coordinate = np.meshgrid(*(spatial_coord[i] for i in range(len(size))), indexing="ij")
    coordinate = np.stack([c for c in coordinate], 0)
    coordinate = np.reshape(coordinate, (dim, -1))
    im = scnd.interpolation.map_coordinates(im, coordinate, order=interp_order)
    im = np.reshape(im, size)
    return im
