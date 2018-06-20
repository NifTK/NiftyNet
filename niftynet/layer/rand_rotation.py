# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import scipy.ndimage

from niftynet.layer.base_layer import RandomisedLayer


class RandomRotationLayer(RandomisedLayer):
    """
    generate randomised rotation matrix for data augmentation
    """

    def __init__(self, name='random_rotation'):
        super(RandomRotationLayer, self).__init__(name=name)
        self._transform = None
        self.min_angle = None
        self.max_angle = None
        self.rotation_angle_x = None
        self.rotation_angle_y = None
        self.rotation_angle_z = None

    def init_uniform_angle(self, rotation_angle=(-10.0, 10.0)):
        assert rotation_angle[0] < rotation_angle[1]
        self.min_angle = float(rotation_angle[0])
        self.max_angle = float(rotation_angle[1])

    def init_non_uniform_angle(self,
                               rotation_angle_x,
                               rotation_angle_y,
                               rotation_angle_z):
        if len(rotation_angle_x):
            assert rotation_angle_x[0] < rotation_angle_x[1]
        if len(rotation_angle_y):
            assert rotation_angle_y[0] < rotation_angle_y[1]
        if len(rotation_angle_z):
            assert rotation_angle_z[0] < rotation_angle_z[1]
        self.rotation_angle_x = [float(e) for e in rotation_angle_x]
        self.rotation_angle_y = [float(e) for e in rotation_angle_y]
        self.rotation_angle_z = [float(e) for e in rotation_angle_z]

    def randomise(self, spatial_rank=3):
        if spatial_rank == 3:
            self._randomise_transformation_3d()
        else:
            # currently not supported spatial rank for rand rotation
            pass

    def _randomise_transformation_3d(self):
        angle_x = 0.0
        angle_y = 0.0
        angle_z = 0.0
        if self.min_angle is None and self.max_angle is None:
            # generate transformation
            if len(self.rotation_angle_x) >= 2:
                angle_x = np.random.uniform(
                    self.rotation_angle_x[0],
                    self.rotation_angle_x[1]) * np.pi / 180.0

            if len(self.rotation_angle_y) >= 2:
                angle_y = np.random.uniform(
                    self.rotation_angle_y[0],
                    self.rotation_angle_y[1]) * np.pi / 180.0

            if len(self.rotation_angle_z) >= 2:
                angle_z = np.random.uniform(
                    self.rotation_angle_z[0],
                    self.rotation_angle_z[1]) * np.pi / 180.0
        else:
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
        if interp_order < 0:
            return image_3d
        assert image_3d.ndim == 3
        assert self._transform is not None
        assert all([dim > 1 for dim in image_3d.shape]), \
            'random rotation supports 3D inputs only'
        center_ = 0.5 * np.asarray(image_3d.shape, dtype=np.int64)
        c_offset = center_ - center_.dot(self._transform)
        image_3d[...] = scipy.ndimage.affine_transform(
            image_3d[...], self._transform.T, c_offset, order=interp_order)
        return image_3d

    def layer_op(self, inputs, interp_orders, *args, **kwargs):
        if inputs is None:
            return inputs

        if isinstance(inputs, dict) and isinstance(interp_orders, dict):
            for (field, image) in inputs.items():
                interp_order = interp_orders[field][0]
                for channel_idx in range(image.shape[-1]):
                    if image.ndim == 4:
                        inputs[field][..., channel_idx] = \
                            self._apply_transformation_3d(
                                image[..., channel_idx], interp_order)
                    elif image.ndim == 5:
                        for t in range(image.shape[-2]):
                            inputs[field][..., t, channel_idx] = \
                                self._apply_transformation_3d(
                                    image[..., t, channel_idx], interp_order)
                    else:
                        raise NotImplementedError("unknown input format")
            # shapes = []
            # for (field, image) in inputs.items():
            #     shapes.append(image.shape)
            # assert(len(shapes) == 2 and shapes[0][0:4] == shapes[1][0:4]), shapes
        else:
            raise NotImplementedError("unknown input format")
        return inputs

        # if inputs.spatial_rank == 3:
        #    if inputs.data.ndim == 4:
        #        for mod_i in range(inputs.data.shape[-1]):
        #            inputs.data[..., mod_i] = self._apply_transformation_3d(
        #                inputs.data[..., mod_i], inputs.interp_order)
        #    if inputs.data.ndim == 5:
        #        for t in range(inputs.data.shape[-1]):
        #            for mod_i in range(inputs.data.shape[-2]):
        #                inputs.data[..., mod_i, t] = \
        #                    self._apply_transformation_3d(
        #                      inputs.data[..., mod_i, t], inputs.interp_order)
        #    if inputs.interp_order > 0:
        #        inputs.data = inputs.data.astype(np.float)
        #    elif inputs.interp_order == 0:
        #        inputs.data = inputs.data.astype(np.int64)
        #    else:
        #        raise ValueError('negative interpolation order')
        #    return inputs
        # else:
        #    # TODO: rotation for spatial_rank is 2
        #    # currently not supported 2/2.5D rand rotation
        #    return inputs
