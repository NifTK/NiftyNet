# -*- coding: utf-8 -*-
"""
To-memory image sink module
"""
from __future__ import absolute_import

from niftynet.io.base_image_sink import BaseImageSink

# Name of the data_param namespace entry for the output callback
MEMORY_OUTPUT_CALLBACK_PARAM = 'output_callback_function'


class MemoryImageSink(BaseImageSink):
    """
    This class enables the writing of result images
    to memory
    """

    def __init__(self,
                 source,
                 interp_order,
                 output_callback_function,
                 name='memory_image_sink'):
        """
        :param output_callback_function: a function accepting an output image
        tensor and an image identifier (str), and the input image tensor from
        which the output was generated.
        """

        super(MemoryImageSink, self).__init__(source, interp_order, name=name)

        self._output_callback_function = output_callback_function

    def layer_op(self, image_data_out, subject_name, image_data_in):
        image_data_out = self._invert_preprocessing(image_data_out)

        self._output_callback_function(image_data_out, subject_name,
                                       image_data_in)
