# -*- coding: utf-8 -*-
import warnings
import scipy.ndimage
import numpy as np
from external.intensity_range_standardization import IntensityRangeStandardization

warnings.simplefilter("ignore", UserWarning)


def rand_window_location_3d(img_size_3d, win_size, n_samples):
    x_start = np.random.random_integers(
        0, np.max((img_size_3d[0] - win_size, 1)), n_samples)
    y_start = np.random.random_integers(
        0, np.max((img_size_3d[1] - win_size, 1)), n_samples)
    z_start = np.random.random_integers(
        0, np.max((img_size_3d[2] - win_size, 1)), n_samples)

    x_end = x_start + win_size
    y_end = y_start + win_size
    z_end = z_start + win_size

    locations = np.stack((x_start, x_end, y_start, y_end, z_start, z_end)).T
    return locations

def grid_window_location_3d(img_size_3d, win_size, grid_size):
    if grid_size <= 0:
        return None
    xs = __enumerate_step_points(
        0, img_size_3d[0], win_size, grid_size)
    ys = __enumerate_step_points(
        0, img_size_3d[1], win_size, grid_size)
    zs = __enumerate_step_points(
        0, img_size_3d[2], win_size, grid_size)
    x_start, y_start, z_start = np.meshgrid(xs, ys, zs)

    x_start = x_start.flatten()
    y_start = y_start.flatten()
    z_start = z_start.flatten()

    x_end = x_start + win_size
    y_end = y_start + win_size
    z_end = z_start + win_size

    locations = np.stack((x_start, x_end, y_start, y_end, z_start, z_end)).T
    return locations

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
        img[:,:,:,mod_i] = scipy.ndimage.affine_transform(
            img[:,:,:,mod_i], transform.T, c_offset, order=3)
    seg = scipy.ndimage.affine_transform(
        seg, transform.T, c_offset, order=0)
    return img.astype(np.float), seg.astype(np.int64)

def rand_spatial_scaling(img, seg=None, percentage=10):
    rand_zoom = (np.random.randint(-percentage, percentage) + 100.0) / 100.0
    img = np.stack([scipy.ndimage.zoom(img[:,:,:,mod_i], rand_zoom, order=3)
                    for mod_i in range(img.shape[-1])], axis=-1)
    seg = scipy.ndimage.zoom(seg, rand_zoom, order=0) \
        if seg is not None else None
    return img, seg

def __enumerate_step_points(starting, ending, win_size, step_size):
    sampling_point_set = []
    while (starting + win_size) <= ending:
        sampling_point_set.append(starting)
        starting = starting + step_size
    sampling_point_set.append(np.max((ending - win_size, 0)))
    return np.unique(sampling_point_set).flatten()
