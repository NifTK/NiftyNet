# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from six.moves import range

# import nn.data_augmentation as dataug
# import utilities.misc as util
# from nn.preprocess import HistNormaliser
from .base import Layer
from .input_placeholders import ImagePatch


class ImageSampler(Layer):
    """
    This class defines a simple example of sampler, it generates
    constant image patches for testing purposes
    """

    def __init__(self, patch, name='sampler'):
        super(ImageSampler, self).__init__(name=name)

        assert isinstance(patch, ImagePatch)
        self.patch = patch
        self.volume_preprocessor = None

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

    @property
    def placeholders(self):
        return self.patch._placeholders

    @property
    def placeholder_names(self):
        names = [placeholder.name.split(':')[0]
                 for placeholder in self.placeholders]
        return tuple(names)

    @property
    def placeholder_dtypes(self):
        dtypes = [placeholder.dtype
                  for placeholder in self.placeholders]
        return tuple(dtypes)

    @property
    def placeholder_shapes(self):
        shapes = [placeholder.get_shape().as_list()
                  for placeholder in self.placeholders]
        return tuple(shapes)
