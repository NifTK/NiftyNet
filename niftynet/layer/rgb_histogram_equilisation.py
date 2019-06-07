from __future__ import absolute_import, print_function

import numpy as np
from niftynet.layer.base_layer import Layer
from niftynet.utilities.util_import import require_module

class RGBHistogramEquilisationLayer(Layer):
    """
    RGB histogram equilisation. Unlike the multi-modality general
    histogram normalisation this is done conventionally, on a
    per-image basis. This layer requires OpenCV.
    """

    def __init__(self,
                 image_name,
                 name='rgb_normaliser'):
        super(RGBHistogramEquilisationLayer, self).__init__(name=name)

        self.image_name = image_name

    def _normalise_image(self, image):
        """
        Normalises a 2D RGB image, if necessary performs any type casting
        and reshaping operations.
        :param image: a 2D RGB image, possibly given as a 5D tensor
        :return: the normalised image in its original shape
        """

        if isinstance(image.dtype, np.floating) and image.dtype != np.float32:
            image = image.astype(np.float32)
        elif isinstance(image.dtype, np.uint):
            image = image.astype(np.float32)/255

        orig_shape = list(image.shape)
        if len(orig_shape) == 5 and (orig_shape[2] > 1 or orig_shape[3] > 1):
            raise ValueError('Can only process 2D images.')

        if len(image.shape) != 3:
            image = image.reshape(orig_shape[:2] + [orig_shape[-1]])

        image = image[...,::-1]

        cv2 = require_module('cv2')
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        intensity = (255*yuv_image[...,0]).astype(np.uint8)
        yuv_image[...,0] = cv2.equalizeHist(intensity).astype(np.float32)/255

        return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)[...,::-1]\
                  .reshape(orig_shape)

    def layer_op(self, image, mask=None):
        """
        :param image: a 3-channel tensor assumed to be an image in floating-point
        RGB format (each channel in [0, 1])
        :return: the equilised image
        """

        if isinstance(image, dict):
            image[self.image_name] = self._normalise_image(
                image[self.image_name])

            return image, mask
        else:
            return self._normalise_image(image), mask
