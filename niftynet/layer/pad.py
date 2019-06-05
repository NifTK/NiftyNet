# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.base_layer import Layer, Invertible


class PadLayer(Layer, Invertible):
    """
    This class defines a padding operation:
    pad `2*border` pixels from spatial dims of the input (numpy array),
    and return the padded input.

    This function is used at volume level (as a preprocessor in image reader)
    therefore assumes the input has at least three spatial dims.
    """

    def __init__(self, image_name, border, name='pad', mode='minimum', pad_to=(0,)):
        """
        :param image_name: the name of the relevant key in the data dictionary
        :param border: the dimensions of the desired border around the image.
        :param name: name of the PadLayer in the tensorflow graph.
        :param mode: how to choose the padding values for the np.pad operation.
        :param pad_to: this determines a desired size of the padded image (useful
            for inconsistent input sizes or for making inference efficient). If
            it == (0, ) (DEFAULT), it will use the constant padding mode
            determined by 'border'
        """
        super(PadLayer, self).__init__(name=name)
        try:
            spatial_border = tuple(map(lambda x: (x,), border))
        except (ValueError, TypeError):
            tf.logging.fatal("Unknown padding param. {}".format(border))
            raise
        self.border = spatial_border
        self.image_name = set(image_name)
        self.mode = mode
        self.pad_to = pad_to
        self.full_border = None

    def layer_op(self, input_image, mask=None):
        if not isinstance(input_image, dict):
            self._set_full_border(input_image)
            input_image = np.pad(input_image, self.full_border, mode=self.mode)
            return input_image, mask

        for name, image in input_image.items():
            self._set_full_border(image)
            if name not in self.image_name:
                tf.logging.warning('could not pad, dict name %s not in %s', name, self.image_name)
                continue
            input_image[name] = np.pad(image, self.full_border, mode=self.mode)
        return input_image, mask

    def inverse_op(self, input_image, mask=None):
        if not isinstance(input_image, dict):
            # you can run the cropping op without running the padding op, but only if you
            # pad with a constant amount (not pad_to)
            if self.full_border is None and self.pad_to == (0,):
                self._set_full_border(input_image)

            outputs = self._crop_numpy_array(input_image, self.full_border)
            return outputs, mask

        for name, image in input_image.items():
            # you can run the cropping op without running the padding op, but only if you
            # pad with a constant amount (not pad_to)
            if self.full_border is None and self.pad_to == (0,):
                self._set_full_border(image)

            if name not in self.image_name:
                continue
            input_image[name] = self._crop_numpy_array(image, self.full_border)
        return input_image, mask

    @staticmethod
    def _crop_numpy_array(image, border):
        try:
            assert image.ndim >= 3, "input image must have at least 3 spatial dims"
            if np.shape(border)[-1] < 2:
                # same amount cropped from each side of array
                border = np.hstack([np.array(border), np.array(border)])

            x_ = border[0][0] if image.shape[0] / 2 > border[0][0] > 0 else 0
            y_ = border[1][0] if image.shape[1] / 2 > border[1][0] > 0 else 0
            z_ = border[2][0] if image.shape[2] / 2 > border[2][0] > 0 else 0
            _x = -border[0][1] if image.shape[0] / 2 > border[0][1] > 0 else image.shape[0]
            _y = -border[1][1] if image.shape[1] / 2 > border[1][1] > 0 else image.shape[1]
            _z = -border[2][1] if image.shape[2] / 2 > border[2][1] > 0 else image.shape[2]
            return image[x_:_x, y_:_y, z_:_z, ...]
        except (IndexError, AssertionError):
            tf.logging.fatal(
                "Unable to invert the padding. Input: {}, pad param. {}".format(image.shape, border))
            raise

    def _set_full_border(self, image):
        """
        To calculate and set the border that is used to a) pad the image and b) invert the padding
        :param image: the input image
        """
        if self.pad_to == (0,):
            full_border = self.border
            while len(full_border) < image.ndim:
                # here, we extend the tuple with zeros as all padding is symmetric (for each
                # dimension, we pad 'in front' and 'behind' with the same number of values).
                full_border = full_border + ((0,),)
        else:
            necessary_padding = np.array(self.pad_to) - image.shape[:len(self.pad_to)]
            full_border = tuple()

            # do not pad if the dimension is bigger than self.pad_to
            necessary_padding[necessary_padding < 0] = 0
            for pad in necessary_padding:
                full_border += ((pad // 2, (pad + 1) // 2),)

            # no padding on the channel dimensions
            while len(full_border) < image.ndim:
                # in pad_to mode, we explicitly determine the padding at the front and back.
                full_border += ((0, 0),)

        self.full_border = full_border
