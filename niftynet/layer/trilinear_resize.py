# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.layer.layer_util import infer_spatial_rank


class TrilinearResizeLayer(Layer):
    """
    Resize 3D volumes using ``tf.image.resize_images``
    (without trainable parameters)
    """

    def __init__(self, size_3d, name='trilinear_resize'):
        """

        :param size_3d: 3-element integers set the output 3D spatial shape
        :param name: layer name string
        """
        super(TrilinearResizeLayer, self).__init__(name=name)
        self.size_3d = size_3d

    def layer_op(self, input_tensor):
        """
        Computes trilinear interpolation using TF ``resize_images`` function.

        :param input_tensor: 3D volume, shape
            ``batch, X, Y, Z, Channels``
        :return: interpolated volume
        """

        assert infer_spatial_rank(input_tensor) == 3, \
            "Trilinear interpolation can only be applied to 3D volumes."
        assert len(self.size_3d) == 3, \
            "Output spatial shape should have 3 integers."

        b_size, x_size, y_size, z_size, c_size = \
            input_tensor.get_shape().as_list()
        x_size_new, y_size_new, z_size_new = self.size_3d

        # resize y-z
        squeeze_b_x = tf.reshape(
            input_tensor, [-1, y_size, z_size, c_size])
        resize_b_x = tf.image.resize_bilinear(
            squeeze_b_x, [y_size_new, z_size_new])
        resume_b_x = tf.reshape(
            resize_b_x, [b_size, x_size, y_size_new, z_size_new, c_size])

        # resize x-y
        #   first reorient
        reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
        #   squeeze and 2d resize
        squeeze_b_z = tf.reshape(
            reoriented, [-1, y_size_new, x_size, c_size])
        resize_b_z = tf.image.resize_bilinear(
            squeeze_b_z, [y_size_new, x_size_new])
        resume_b_z = tf.reshape(
            resize_b_z, [b_size, z_size_new, y_size_new, x_size_new, c_size])

        return tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
