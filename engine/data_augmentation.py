# -*- coding: utf-8 -*-
import warnings
import scipy.ndimage
import numpy as np

warnings.simplefilter("ignore", UserWarning)


def rand_rotation_3d(img, seg, max_angle=10):
    # generate transformation
    angle_x = np.random.randint(-max_angle, max_angle) * np.pi / 180.0
    angle_y = np.random.randint(-max_angle, max_angle) * np.pi / 180.0
    angle_z = np.random.randint(-max_angle, max_angle) * np.pi / 180.0
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

    center_ = 0.5 * np.asarray(img.shape[:-1], dtype=np.int64)
    c_offset = center_ - center_.dot(transform)
    # apply transformation to each volume
    for mod_i in range(img.shape[-1]):
        img[:, :, :, mod_i] = scipy.ndimage.affine_transform(
            img[:, :, :, mod_i], transform.T, c_offset, order=3)
    seg = scipy.ndimage.affine_transform(
        seg, transform.T, c_offset, order=0)
    return img.astype(np.float), seg.astype(np.int64)


def rand_biasfield_3d(shape, max_range, pixdim=(1, 1, 1), order=3):
    bias_field = np.zeros(shape)
    x_p = np.arange(0, shape[0]*pixdim[0], step=pixdim[0])/(shape[0]*pixdim[0])-0.5
    y_p = np.arange(0, shape[1]*pixdim[1], step=pixdim[1])/(shape[1]*pixdim[1])-0.5
    z_p = np.arange(0, shape[2] * pixdim[2], step=pixdim[2]) / (shape[2] * pixdim[2]) - 0.5
    meshgrid = np.vstack(np.meshgrid(x_p, y_p, z_p)).reshape(3, -1).T
    for x_i in range(0, order+1):
        order_fin = x_i
        for y_i in range(0, order+1-order_fin):
            order_fin = y_i + order_fin
            for z_i in range(0, order+1-order_fin):
                random_coeff = (2*max_range)*np.random.ranf()-max_range
                function_bias = random_coeff * (np.power(meshgrid[:, 0], x_i) + np.power(meshgrid[:, 1], y_i) + np.power(meshgrid[:, 2], z_i))
                function_to_add = np.reshape(function_bias, shape)
                bias_field = bias_field + function_to_add
    return np.exp(bias_field)


def apply_rand_biasfield(img, max_range, pixdim=(1, 1, 1), order=3):
    shape = img.shape
    if img.ndim == 3:
        bias_field = rand_biasfield_3d(shape, max_range=max_range, pixdim=pixdim, order=order)
        return img * bias_field
    if img.ndim == 4:
        bias_field = np.zeros(shape)
        for n in range(0,shape[3]):
            bias_field[:, :, :, n] = rand_biasfield_3d(shape[0:3], max_range=max_range, pixdim=pixdim, order=order)
        return img * bias_field


def rand_spatial_scaling(img, seg=None, percentage=10):
    rand_zoom = (np.random.randint(-percentage, percentage) + 100.0) / 100.0
    img = np.stack([scipy.ndimage.zoom(img[:,:,:,mod_i], rand_zoom, order=3)
                    for mod_i in range(img.shape[-1])], axis=-1)
    seg = scipy.ndimage.zoom(seg, rand_zoom, order=0) \
        if seg is not None else None
    return img, seg

