# -*- coding: utf-8 -*-
"""
windows aggregator resize each item
in a batch output and save as an image
"""
from __future__ import absolute_import, division, print_function

import os

import numpy as np
from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator
from niftynet.io.file_image_sink import BaseFileImageSink
from niftynet.io.file_image_sink_meta_decorator import FileImageSinkDecorator


class _CSVWriterDecorator(FileImageSinkDecorator):
    """
    Decorator that logs classification results
    in a CSV file.
    """

    def __init__(self, base_writer):
        """
        :param base_writer: Underlying image writer instance
        """

        super(_CSVWriterDecorator, self).__init__(base_writer)

        self.csv_path = os.path.join(base_writer.output_path,
                                     base_writer.postfix + '.csv')
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    # pylint: disable=arguments-differ
    def layer_op(self, image_out, subject_name, image_in):
        super(_CSVWriterDecorator, self).layer_op(image_out, subject_name,
                                                  image_in)

        with open(self.csv_path, 'a') as csv_file:
            data_str = ','.join([str(i) for i in image_out[0, 0, 0, 0, :]])
            csv_file.write(subject_name + ',' + data_str + '\n')


class ClassifierSamplesAggregator(ImageWindowsAggregator):
    """
    This class decodes each item in a batch by saving classification
    labels to a new image volume.
    """

    def __init__(self, image_reader, image_writer, name='image'):
        if isinstance(image_writer, BaseFileImageSink):
            image_writer = _CSVWriterDecorator(image_writer)

        ImageWindowsAggregator.__init__(
            self, image_reader, image_writer, name=name)
        self.name = name
        self.output_interp_order = 0

    def decode_batch(self, window, location):
        """
        window holds the classifier labels
        location is a holdover from segmentation and may be removed
        in a later refactoring, but currently hold info about the stopping
        signal from the sampler
        """
        n_samples = window.shape[0]
        for batch_id in range(n_samples):
            if self._is_stopping_signal(location[batch_id]):
                return False
            self.image_id = location[batch_id, 0]
            self._save_current_image(window[batch_id, ...])
        return True

    def _save_current_image(self, image_out):
        if self.input_image is None:
            return
        window_shape = [1, 1, 1, 1, image_out.shape[-1]]
        image_out = np.reshape(image_out, window_shape)

        self.writer(image_out, self.image_id, self.input_image[self.name])
