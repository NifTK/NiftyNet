# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer.base_layer import Layer
from niftynet.utilities.input_placeholders import ImagePatch


class BaseSampler(Layer):
    """
    This class defines the basic operations of sampling
    to generate image patches for training/inference

    The layer_op should return an iterable object
    that yields an instance of ImagePatch

    please see the example toy_sampler.py
    """

    def __init__(self, patch, name='sampler'):
        super(BaseSampler, self).__init__(name=name)

        assert isinstance(patch, ImagePatch)
        self.patch = patch
        self._placeholders = self.patch.create_placeholders()

    def layer_op(self, batch_size):
        """
        should return an input placeholder, i.e.:
        yield self.patch
        """
        raise NotImplementedError

    @property
    def placeholders(self):
        # This is required to connect the sampler to an input buffer
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
