# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.utilities.util_common import look_up_operations

SUPPORTED_OPS = set(["SOFTMAX", "ARGMAX", "IDENTITY"])


class PostProcessingLayer(Layer):
    """
    This layer operation converts the raw network outputs into final inference
    results.
    """

    def __init__(self, func='', num_classes=0, name='post_processing'):
        super(PostProcessingLayer, self).__init__(name=name)
        self.func = look_up_operations(func.upper(), SUPPORTED_OPS)
        self.num_classes = num_classes

    def num_output_channels(self):
        assert self._op._variables_created
        if self.func == "SOFTMAX":
            return self.num_classes
        else:
            return 1

    def layer_op(self, inputs):
        if self.func == "SOFTMAX":
            output_tensor = tf.cast(tf.nn.softmax(inputs), tf.float32)
        elif self.func == "ARGMAX":
            output_tensor = tf.cast(tf.argmax(inputs, -1), tf.int32)
            output_tensor = tf.expand_dims(output_tensor, axis=-1)
        elif self.func == "IDENTITY":
            output_tensor = tf.cast(inputs, tf.float32)
        return output_tensor
