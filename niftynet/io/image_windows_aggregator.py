# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import numpy as np

class ImageWindowsAggregator(object):

    def __init__(self, image_reader=None):
        self.reader = image_reader

    def decode_batch(self, *args, **kwargs):
        raise NotImplementedError


class GridSamplesAggregator(ImageWindowsAggregator):
    def __init__(self, image_reader):
        ImageWindowsAggregator.__init__(self, image_reader=image_reader)
        self.current_id = None
        self.image_out = None

    def decode_batch(self, window, location):
        n_samples = location.shape[0]

        for batch_id in range(n_samples):
            image_id = location[batch_id, 0]
            if image_id != self.current_id:
                if self.image_out is not None:
                    self._save_image()
                self.current_id = image_id
                self.image_out = self._create_empty_image(window.shape)
            #TODO: put window to image_out
        import pdb; pdb.set_trace()
        return

    def _create_empty_image(self, window_shape):
        spatial_shape = \
            self.reader.output_list[self.current_id]['image'].shape[:3]
        n_channels = window_shape[-1]
        output_image_shape = spatial_shape + (n_channels,)
        # TODO: define dtype of output
        empty_image = np.zeros(output_image_shape)
        import pdb; pdb.set_trace()
        return empty_image

    def _save_image(self):
        self.image_out
        import pdb; pdb.set_trace()
        pass