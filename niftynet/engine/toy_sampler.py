# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
from six.moves import range

from niftynet.engine.base_sampler import BaseSampler


class ToySampler(BaseSampler):
    """
    This class defines a simple example of sampler, it generates
    constant image patches for testing purposes
    """

    def __init__(self, patch, name='toy_sampler'):
        super(ToySampler, self).__init__(patch=patch, name=name)

    def layer_op(self, batch_size=1):
        # batch_size is needed here so that it generates total number of
        # N samples where (N % batch_size) == 0
        n_item = 1
        all_item = ((n_item // batch_size) + 1) * batch_size
        for i in range(all_item):

            # generate an image
            self.patch.image = np.ones(self.patch.full_image_shape)

            # generate location information
            info = np.zeros(self.patch.full_info_shape)
            info[0] = i
            self.patch.info = info

            # generate a label
            if self.patch.has_labels:
                self.patch.label = np.zeros(
                    self.patch.full_label_shape)

            # generate a weight map
            if self.patch.has_weight_maps:
                self.patch.weight_map = np.zeros(
                    self.patch.full_weight_map_shape)

            yield self.patch
