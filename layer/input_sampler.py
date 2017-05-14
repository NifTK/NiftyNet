# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from six.moves import range

#import nn.data_augmentation as dataug
#import utilities.misc as util
#from nn.preprocess import HistNormaliser
from .base import Layer


class ImageSampler(Layer):
    """
    This class assumes all images have same length in all spatial dims
    i.e., image_shape = [image_size] * spatial_rank
    and full_image_size = [image_spatial_shape] + [number_of_modalities]
    """

    def __init__(self,
                 image_shape,
                 label_shape=None,
                 image_dtype=tf.float32,
                 label_dtype=tf.int64,
                 spatial_rank=3,
                 num_modality=1,
                 name='sampler'):
        assert len(set(image_shape)) == 1
        if label_shape is not None:
            assert len(set(label_shape)) == 1
        super(ImageSampler, self).__init__(name=name)

        self._image_shape = image_shape
        self._label_shape = label_shape
        self._image_dtype = image_dtype
        self._label_dtype = label_dtype

        # spatial_rank == 3 for volumetric image
        self._spatial_rank = spatial_rank
        self._num_modality = num_modality

        with self._enter_variable_scope():
            self._placeholders = self._create_placeholders()
        self.volume_preprocessor = None

    @property
    def image_size(self):
        # assumes the samples have the same length in all spatial dims
        return set(self._image_shape).pop()

    @property
    def label_size(self):
        if self._label_shape is not None:
            # assumes the samples have the same length in all spatial dims
            return set(self._label_shape).pop()
        return None

    @property
    def full_image_shape(self):
        return [self.image_size] * self._spatial_rank + [self._num_modality]

    @property
    def full_label_shape(self):
        if self._label_shape is not None:
            # assumes the samples have the same length in all spatial dims
            return [self.label_size] * self._spatial_rank
        return None

    @property
    def full_info_shape(self):
        """
        `info` contains the spatial location of a image patch
        it will be used to put the sampled patch back to the original volume
        the first dim: volume id
        the size of the other dims: spatial_rank * 2, indicating starting
        and end point of a patch in each dim
        """
        return [1 + self._spatial_rank * 2]

    def _create_placeholders(self):
        """
        The placeholders are defined so that the input buffer knows how
        to initialise an input queue
        """
        image_placeholders = tf.placeholder(dtype=self._image_dtype,
                                            shape=self.full_image_shape,
                                            name='images')
        location_info_dtype = tf.int64
        info_placeholders = tf.placeholder(dtype=location_info_dtype,
                                           shape=self.full_info_shape,
                                           name='info')
        if self._label_shape is not None:
            label_placeholders = tf.placeholder(dtype=self._label_dtype,
                                                shape=self.full_label_shape,
                                                name='labels')
            return (image_placeholders, label_placeholders, info_placeholders)
        else:
            return (image_placeholders, info_placeholders)

    @property
    def placeholders(self):
        return self._placeholders

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

    def layer_op(self, batch_size=1):
        # batch_size is needed here so that it generates total number of
        # N samples where (N % batch_size) == 0
        i = 0
        n_item = 4
        all_item = ((n_item / batch_size) + 1) * batch_size
        for i in range(all_item):
            images = np.ones(self.full_image_shape)
            info = np.zeros(self.full_info_shape)
            info[0] = i
            if self.full_label_shape is not None:
                labels = np.zeros(self.full_label_shape)
                output_tuple = (images, labels, info)
            else:
                output_tuple = (images, info)
            yield {self.placeholders: output_tuple}
