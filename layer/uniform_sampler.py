# -*- coding: utf-8 -*-
import numpy as np
from six.moves import range

from .base_sampler import BaseSampler


class UniformSampler(BaseSampler):
    """
    This class generators samples by uniformly sampling each input volume
    """

    def __init__(self, patch, name="uniform_sampler"):
        super(UniformSampler, self).__init__(patch=patch, name=name)

    def layer_op(self, batch_size=1):
        # batch_size is needed here so that it generates total number of
        # N samples where (N % batch_size) == 0
        i = 0
        n_item = 4
        all_item = ((n_item / batch_size) + 1) * batch_size
        for i in range(all_item):
            out_list = []

            # generate an image
            images = np.ones(self.patch.full_image_shape)
            out_list.append(images)

            # generate location information
            info = np.zeros(self.patch.full_info_shape)
            info[0] = i
            out_list.append(info)

            # generate a label
            if self.patch.has_labels:
                labels = np.zeros(self.patch.full_label_shape)
                out_list.append(labels)

            # generate a weight map
            if self.patch.has_weight_maps:
                weight_maps = np.zeros(self.patch.full_weight_map_shape)
                out_list.append(weight_maps)

            yield {self.placeholders: tuple(out_list)}
