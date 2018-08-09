# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import scipy.ndimage
import nibabel as nib

from niftynet.layer.base_layer import RandomisedLayer


class RandomBiasFieldLayer(RandomisedLayer):
    """
    generate randomised bias field transformation for data augmentation
    """

    def __init__(self, name='random_bias_field'):
        super(RandomBiasFieldLayer, self).__init__(name=name)
        self._bf_coeffs = None
        self.min_coeff = None
        self.max_coeff = None
        self.order = None

    def init_uniform_coeff(self, coeff_range=(-10.0, 10.0)):
        assert coeff_range[0] < coeff_range[1]
        self.min_coeff = float(coeff_range[0])
        self.max_coeff = float(coeff_range[1])

    def init_order(self, order=3):
        self.order = int(order)

    def randomise(self, spatial_rank=3):
        self._generate_bias_field_coeffs(spatial_rank)

    def _generate_bias_field_coeffs(self, spatial_rank):
        '''
        Sampling of the appropriate number of coefficients for the creation
        of the bias field map
        :param spatial_rank: spatial rank of the image to modify
        :return:
        '''
        rand_coeffs = []
        if spatial_rank == 3:
            for order_x in range(0, self.order+1):
                for order_y in range(0, self.order +1- order_x):
                    for order_z in range(0, self.order+1 -(order_x + order_y)):
                        rand_coeff_new = np.random.uniform(self.min_coeff,
                                                       self.max_coeff)
                        rand_coeffs.append(rand_coeff_new)
        else:
            for order_x in range(0, self.order+1):
                for order_y in range(0, self.order +1- order_x):
                    rand_coeff_new = np.random.uniform(self.min_coeff,
                                                       self.max_coeff)
                    rand_coeffs.append(rand_coeff_new)
        self._bf_coeffs = rand_coeffs

    def _generate_bias_field_map(self, shape):
        '''
        Create the bias field map using a linear combination polynomial
        functions and the coefficients previously sampled
        :param shape: shape of the image in order to create the polynomial
        functions
        :return: bias field map to apply
        '''
        spatial_rank = len(shape)
        x_range = np.arange(-shape[0] / 2, shape[0] / 2)
        y_range = np.arange(-shape[1] / 2, shape[1] / 2)
        bf_map = np.zeros(shape)
        i = 0
        if spatial_rank == 3:
            z_range = np.arange(-shape[2] / 2, shape[2] / 2)
            x_mesh, y_mesh, z_mesh = np.asarray(np.meshgrid(x_range, y_range,
                                                       z_range), dtype=float)
            x_mesh /= float(np.max(x_mesh))
            y_mesh /= float(np.max(y_mesh))
            z_mesh /= float(np.max(z_mesh))
            for order_x in range(0, self.order+1):
                for order_y in range(0, self.order + 1 - order_x):
                    for order_z in range(0, self.order + 1 - (order_x+order_y)):
                        rand_coeff = self._bf_coeffs[i]

                        new_map = rand_coeff * np.power(x_mesh, order_x) * \
                                            np.power(y_mesh, order_y) * \
                                  np.power(z_mesh, order_z)
                        # print(np.asarray(np.where(np.abs(new_map) >
                        #                           0)).shape, np.unique(
                        #     new_map).shape)
                        bf_map += np.transpose(new_map, (1, 0, 2))
                        i += 1
        if spatial_rank == 2:
            x_mesh, y_mesh = np.asarray(np.meshgrid(x_range, y_range),
                                        dtype=float)
            x_mesh /= np.max(x_mesh)
            y_mesh /= np.max(y_mesh)
            for order_x in range(0, self.order+1):
                for order_y in range(0, self.order+1 - order_x):
                    rand_coeff = self._bf_coeffs[i]
                    new_map = rand_coeff * np.power(x_mesh, order_x) * \
                                            np.power(y_mesh, order_y)
                    bf_map += np.transpose(new_map, (1, 0))
                    i += 1
        return np.exp(bf_map)

    def _apply_transformation(self, image):
        '''
        Create the bias field map based on the randomly sampled coefficients
        and apply it (
        multiplicative) to the image
        to augment
        :param image image on which to apply the bias field augmentation:
        :return: modified image
        '''
        assert self._bf_coeffs is not None
        bf_map = self._generate_bias_field_map(image.shape)
        print(np.asarray(np.where(np.abs(bf_map)>0)).shape)
        bf_image = image * bf_map
        print(bf_image.shape, image.shape, np.max(bf_image), np.max(image),
              np.max(bf_map), np.min(bf_map))
        bf_nii = nib.Nifti1Image(bf_map, np.diag([1,1,1,1]))
        bf_image_nii = nib.Nifti1Image(bf_image, np.diag([1,1,1,1]))
        image_nii = nib.Nifti1Image(image, np.diag([1,1,1,1]))
        nib.save(bf_nii, './TestBF.nii.gz')
        nib.save(bf_image_nii, './TestModif.nii.gz')
        nib.save(image_nii, './InitModify.nii.gz')
        return bf_image

    def layer_op(self, inputs, interp_orders, *args, **kwargs):
        if inputs is None:
            return inputs
        for (field, image) in inputs.items():
            print(field)
            if field == 'image':
                for mod_i in range(image.shape[-1]):
                    if image.ndim == 4:
                        inputs[field][..., mod_i] = \
                            self._apply_transformation(
                                image[..., mod_i])
                    elif image.ndim == 5:
                        for t in range(image.shape[-2]):
                            inputs[field][..., t, mod_i] = \
                                self._apply_transformation(
                                    image[..., t, mod_i])
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
