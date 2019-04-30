# -*- coding: utf-8 -*-
"""
Image output module
"""
from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

from niftynet.layer.base_layer import Layer
from niftynet.layer.pad import PadLayer
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer

class BaseImageSink(Layer):
    """
    Base class for passthrough layers that write images.
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 source,
                 interp_order,
                 name='image_sink'):
        """
        :param source: the image source of the input images
        for which this layer is to write the outputs.
        :param interp_order: polynomial order of the interpolation applied
        where needed on output.
        """

        super(BaseImageSink, self).__init__(name=name)

        self._source = source
        self.interp_order = interp_order

    @property
    def source(self):
        """
        :return: the source of the images written by this layer
        """

        return self._source

    def _invert_preprocessing(self, image_out):
        """
        Applies the inverse of the pre-processing operations
        applied by the image source.
        :param image_out: the aggregated output samples as a image tensor.
        """

        for layer in reversed(self.source.preprocessors):
            if isinstance(layer, PadLayer):
                image_out, _ = layer.inverse_op(image_out)
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                image_out, _ = layer.inverse_op(image_out)

        return image_out

    # pylint: disable=arguments-differ
    @abstractmethod
    def layer_op(self, image_data_out, subject_name, image_data_in):
        """
        :param image_data_out: the voxel data to output
        :param subject_name: a unique identifier for the subject for which
        the output was generated.
        :param image_data_in: the image object from which the output
        was generated
        """

        return
