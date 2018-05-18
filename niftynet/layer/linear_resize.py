# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.layer.layer_util import expand_spatial_params
from niftynet.layer.layer_util import infer_spatial_rank


class LinearResizeLayer(Layer):
    """
    Resize 2D/3D images using ``tf.image.resize_bilinear``
    (without trainable parameters).
    """

    def __init__(self, new_size, name='trilinear_resize'):
        """

        :param new_size: integer or a list of integers set the output
            2D/3D spatial shape.  If the parameter is an integer ``d``,
            it'll be expanded to ``(d, d)`` and ``(d, d, d)`` for 2D and
            3D inputs respectively.
        :param name: layer name string
        """
        super(LinearResizeLayer, self).__init__(name=name)
        self.new_size = new_size

    def layer_op(self, input_tensor):
        """
        Resize the image by linearly interpolating the input
        using TF ``resize_bilinear`` function.

        :param input_tensor: 2D/3D image tensor, with shape:
            ``batch, X, Y, [Z,] Channels``
        :return: interpolated volume
        """

        input_spatial_rank = infer_spatial_rank(input_tensor)
        assert input_spatial_rank in (2, 3), \
            "linearly interpolation layer can only be applied to " \
            "2D/3D images (4D or 5D tensor)."
        self.new_size = expand_spatial_params(self.new_size, input_spatial_rank)

        if input_spatial_rank == 2:
            return tf.image.resize_bilinear(input_tensor, self.new_size)

        b_size, x_size, y_size, z_size, c_size = \
            input_tensor.shape.as_list()
        x_size_new, y_size_new, z_size_new = self.new_size

        if (x_size == x_size_new) and (y_size == y_size_new) and (
                z_size == z_size_new):
            # already in the target shape
            return input_tensor

        # resize y-z
        squeeze_b_x = tf.reshape(
            input_tensor, [-1, y_size, z_size, c_size])
        resize_b_x = tf.image.resize_bilinear(
            squeeze_b_x, [y_size_new, z_size_new])
        resume_b_x = tf.reshape(
            resize_b_x, [b_size, x_size, y_size_new, z_size_new, c_size])

        # resize x
        #   first reorient
        reoriented = tf.transpose(resume_b_x, [0, 3, 2, 1, 4])
        #   squeeze and 2d resize
        squeeze_b_z = tf.reshape(
            reoriented, [-1, y_size_new, x_size, c_size])
        resize_b_z = tf.image.resize_bilinear(
            squeeze_b_z, [y_size_new, x_size_new])
        resume_b_z = tf.reshape(
            resize_b_z, [b_size, z_size_new, y_size_new, x_size_new, c_size])

        output_tensor = tf.transpose(resume_b_z, [0, 3, 2, 1, 4])
        return output_tensor
