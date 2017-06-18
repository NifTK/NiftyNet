# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from .base_layer import Layer


class ReshapeLayer(Layer):
    """
    This class defines a simple reshape layer, principally for passing feature maps to fully connected layers.
    """

    def __init__(self,
                 output_size,
                 name='reshape'):
        super(ReshapeLayer, self).__init__(name=name)
        self.output_size = output_size

    def layer_op(self, input_tensor):

        output_tensor = tf.reshape(tensor=input_tensor,
                                   shape=self.output_size,
                                   name='reshape')
        return output_tensor
