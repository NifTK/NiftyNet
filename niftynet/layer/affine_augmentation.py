# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.layer.grid_warper import AffineGridWarperLayer
from niftynet.layer.layer_util import infer_spatial_rank
from niftynet.layer.resampler import ResamplerLayer


class AffineAugmentationLayer(Layer):
    """
    This layer applies a small random (per-iteration) affine
    transformation to an image. The distribution of transformations
    generally results in scaling the image up, with minimal sampling
    outside the original image.
    """

    def __init__(self,
                 scale,
                 interpolation='linear',
                 boundary='zero',
                 transform=None,
                 name='AffineAugmentation'):
        """

        :param scale: how extreme the perturbation is, with 0. meaning
            no perturbation and 1.0 giving largest perturbations.
        :param interpolation: the image value interpolation used by
            the resampling.
        :param boundary: the boundary handling used by the resampling
        :param name: string name of the layer.
        """
        Layer.__init__(self, name=name)

        self.scale = min(max(float(scale), 0.0), 1.0)
        self.interpolation = interpolation
        self.boundary = boundary

        self._transform = None
        if transform is not None:
            self._transform = transform

    def _random_transform(self, batch_size, spatial_rank):
        """
        computes a relative transformation
        mapping <-1..1, -1..1, -1..1> to <-1..1, -1..1, -1..1> (in 3D)
        or <-1..1, -1..1> to <-1..1, -1..1> (in 2D).

        :param batch_size: number of different random transformations
        :param spatial_rank: number of spatial dimensions
        :return:
        """
        output_corners = get_relative_corners(spatial_rank)
        output_corners = tf.tile([output_corners], [batch_size, 1, 1])

        # make randomised output corners
        random_size = [batch_size, 2 ** spatial_rank, spatial_rank]
        random_scale = tf.random_uniform(random_size, 1. - self.scale, 1.0)
        source_corners = output_corners * random_scale

        # make homogeneous corners
        batch_ones = tf.ones_like(output_corners[..., 0:1])
        source_corners = tf.concat([source_corners, batch_ones], -1)
        output_corners = tf.concat([output_corners, batch_ones], -1)

        ls_transform = tf.matrix_solve_ls(output_corners, source_corners)
        return tf.transpose(ls_transform, [0, 2, 1])

    def layer_op(self, input_tensor):
        input_shape = input_tensor.shape.as_list()
        batch_size = input_shape[0]
        spatial_shape = input_shape[1:-1]
        spatial_rank = infer_spatial_rank(input_tensor)

        if self._transform is None:
            relative_transform = self._random_transform(
                batch_size, spatial_rank)
            self._transform = relative_transform
        else:
            relative_transform = self._transform

        grid_warper = AffineGridWarperLayer(spatial_shape, spatial_shape)
        resampler = ResamplerLayer(
            interpolation=self.interpolation, boundary=self.boundary)
        warp_parameters = tf.reshape(
            relative_transform[:, :spatial_rank, :], [batch_size, -1])
        grid = grid_warper(warp_parameters)
        resampled = resampler(input_tensor, grid)
        return resampled

    def inverse(self, interpolation=None, boundary=None, name=None):
        """
        create a new layer that will apply the inversed version of
        self._transform. This function write this instance members.
        (calling `self()` after `self.inverse()` might give unexpected results.)

        :param interpolation:
        :param boundary:
        :param name:
        :return: a niftynet layer that inverses the transformation of  `self`.
        """
        if interpolation is None:
            interpolation = self.interpolation
        if boundary is None:
            boundary = self.boundary
        if name is None:
            name = self.name + '_inverse'

        inverse_layer = AffineAugmentationLayer(
            self.scale,
            interpolation,
            boundary,
            tf.matrix_inverse(self._transform),
            name)
        return inverse_layer


def get_relative_corners(spatial_rank):
    """
    compute relative corners of the spatially n-d tensor::

        1-D: [[-1], [1]]
        2-D: [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        3-D: [[-1, -1, -1], [-1, -1, 1],
              [-1, 1, -1],  [-1, 1, 1],
              [1, -1, -1],  [1, -1, 1],
              [1, 1, -1],   [1, 1, 1]]

    :param spatial_rank: integer of number of spatial dimensions
    :return: [2**spatial_rank, spatial_rank] matrix
    """
    return [
        [int(c) * 2.0 - 1.0 for c in format(i, '0%ib' % spatial_rank)]
        for i in range(2 ** spatial_rank)]
