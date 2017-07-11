# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np
import scipy.ndimage

from niftynet.layer.base_layer import Layer


class RandomRotationLayer(Layer):
    """
    generate randomised rotation matrix for data augmentation
    """

    def __init__(self,
                 min_angle=-10.0,
                 max_angle=10.0,
                 name='random_rotation'):
        super(RandomRotationLayer, self).__init__(name=name)
        assert min_angle < max_angle
        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)
        self._transform = None

    def randomise(self, spatial_rank=3):
        if spatial_rank == 3:
            self._randomise_transformation_3d()
        else:
            #currently not supported spatial rank for rand rotation
            pass

    def _randomise_transformation_3d(self):
        # generate transformation
        angle_x = np.random.uniform(
                self.min_angle, self.max_angle) * np.pi / 180.0
        angle_y = np.random.uniform(
                self.min_angle, self.max_angle) * np.pi / 180.0
        angle_z = np.random.uniform(
                self.min_angle, self.max_angle) * np.pi / 180.0
        transform_x = np.array([[np.cos(angle_x), -np.sin(angle_x), 0.0],
                                [np.sin(angle_x), np.cos(angle_x), 0.0],
                                [0.0, 0.0, 1.0]])
        transform_y = np.array([[np.cos(angle_y), 0.0, np.sin(angle_y)],
                                [0.0, 1.0, 0.0],
                                [-np.sin(angle_y), 0.0, np.cos(angle_y)]])
        transform_z = np.array([[1.0, 0.0, 0.0],
                                [0.0, np.cos(angle_z), -np.sin(angle_z)],
                                [0.0, np.sin(angle_z), np.cos(angle_z)]])
        transform = np.dot(transform_z, np.dot(transform_x, transform_y))
        self._transform = transform

    def _apply_transformation_3d(self, image_3d, interp_order=3):
        assert image_3d.ndim == 3
        assert self._transform is not None
        center_ = 0.5 * np.asarray(image_3d.shape, dtype=np.int64)
        c_offset = center_ - center_.dot(self._transform)
        image_3d[:, :, :] = scipy.ndimage.affine_transform(
            image_3d[:, :, :], self._transform.T, c_offset, order=interp_order)
        return image_3d

    def layer_op(self, inputs):
        if inputs is None:
            return inputs
        if inputs.spatial_rank == 3:
            if inputs.data.ndim == 4:
                for mod_i in range(inputs.data.shape[-1]):
                    inputs.data[..., mod_i] = self._apply_transformation_3d(
                        inputs.data[..., mod_i], inputs.interp_order)
            if inputs.data.ndim == 5:
                for t in range(inputs.data.shape[-1]):
                    for mod_i in range(inputs.data.shape[-2]):
                        inputs.data[..., mod_i, t] = \
                            self._apply_transformation_3d(
                                inputs.data[..., mod_i, t], inputs.interp_order)
            if inputs.interp_order > 0:
                inputs.data = inputs.data.astype(np.float)
            elif inputs.interp_order == 0:
                inputs.data = inputs.data.astype(np.int64)
            else:
                raise ValueError('negative interpolation order')
            return inputs
        else:
            # TODO: rotation for spatial_rank is 2
            # currently not supported 2/2.5D rand rotation
            return inputs
