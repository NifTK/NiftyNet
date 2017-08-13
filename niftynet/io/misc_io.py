# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import warnings
import tensorflow as tf

import nibabel as nib
import numpy as np
import scipy.ndimage

image_loaders = [nib.load]
try:
    import niftynet.utilities.simple_itk_as_nibabel

    image_loaders.append(
        niftynet.utilities.simple_itk_as_nibabel.SimpleITKAsNibabel)
except ImportError:
    warnings.warn(
        'SimpleITK adapter failed to load, reducing the supported file formats.',
        ImportWarning)

warnings.simplefilter("ignore", UserWarning)

FILE_EXTENSIONS = [".nii.gz", ".tar.gz"]


#### utilities for file headers

def infer_ndims_from_file(file_path):
    image_header = load_image(file_path).header
    return int(image_header['dim'][0])


def create_affine_pixdim(affine, pixdim):
    '''
    Given an existing affine transformation and the pixel dimension to apply,
    create a new affine matrix that satisfies the new pixel dimension
    :param affine: original affine matrix
    :param pixdim: pixel dimensions to apply
    :return:
    '''
    norm_affine = np.sqrt(np.sum(np.square(affine[:, 0:3]), 0))
    to_divide = np.tile(
        np.expand_dims(np.append(norm_affine, 1), axis=1), [1, 4])
    to_multiply = np.tile(
        np.expand_dims(np.append(np.asarray(pixdim), 1), axis=1), [1, 4])
    return np.multiply(np.divide(affine, to_divide.T), to_multiply.T)


def load_image(filename):
    # load an image from a supported filetype and return an object
    # that matches nibabel's spatialimages interface
    for image_loader in image_loaders:
        try:
            img = image_loader(filename)
            img = correct_image_if_necessary(img)
            return img
        except nib.filebasedimages.ImageFileError:
            # if the image_loader cannot handle the type continue to next loader
            pass
    raise nib.filebasedimages.ImageFileError(
        'No loader could load the file')  # Throw last error


def correct_image_if_necessary(img):
    if img.header['dim'][0] == 5:
        # do nothing for high-dimensional array
        return img
    # Check that affine matches zooms
    pixdim = img.header.get_zooms()
    if not np.array_equal(np.sqrt(np.sum(np.square(img.affine[0:3, 0:3]), 0)),
                          np.asarray(pixdim)):
        if hasattr(img, 'get_sform'):
            # assume it is a malformed NIfTI and try to fix it
            img = rectify_header_sform_qform(img)
    return img


def rectify_header_sform_qform(img_nii):
    '''
    Look at the sform and qform of the nifti object and correct it if any
    incompatibilities with pixel dimensions
    :param img_nii:
    :return:
    '''
    # TODO: check img_nii is a nibabel object
    pixdim = img_nii.header.get_zooms()
    sform = img_nii.get_sform()
    qform = img_nii.get_qform()
    norm_sform = np.sqrt(np.sum(np.square(sform[0:3, 0:3]), 0))
    norm_qform = np.sqrt(np.sum(np.square(qform[0:3, 0:3]), 0))
    flag_sform_problem = False
    flag_qform_problem = False
    if not np.array_equal(norm_sform, np.asarray(pixdim)):
        flag_sform_problem = True
    if not np.array_equal(norm_qform, np.asarray(pixdim)):
        flag_qform_problem = True

    if img_nii.header['sform_code'] > 0:
        if not flag_sform_problem:
            return img_nii
        elif not flag_qform_problem:
            # recover by copying the qform over the sform
            img_nii.set_sform(np.copy(img_nii.get_qform()))
            return img_nii
    elif img_nii.header['qform_code'] > 0:
        if not flag_qform_problem:
            return img_nii
        elif not flag_sform_problem:
            # recover by copying the sform over the qform
            img_nii.set_qform(np.copy(img_nii.get_sform()))
            return img_nii
    affine = img_nii.affine
    pixdim = img_nii.header.get_zooms()[:3]  # TODO: assuming 3 elements
    new_affine = create_affine_pixdim(affine, pixdim)
    img_nii.set_sform(new_affine)
    img_nii.set_qform(new_affine)
    return img_nii


#### end of utilities for file headers


### resample/reorientation original volumes
# Perform the reorientation to ornt_fin of the data array given ornt_init
def do_reorientation(data_array, init_axcodes, final_axcodes):
    '''
    Performs the reorientation (changing order of axes)
    :param data_array: Array to reorient
    :param ornt_init: Initial orientation
    :param ornt_fin: Target orientation
    :return data_reoriented: New data array in its reoriented form
    '''
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array
    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)
    data_reoriented = nib.orientations.apply_orientation(
        data_array, ornt_transf)
    return data_reoriented


# Perform the resampling of the data array given the initial and final pixel
# dimensions and the interpolation order
# this function assumes the same interp_order for multi-modal images
# do we need separate interp_order for each modality?
def do_resampling(data_array, pixdim_init, pixdim_fin, interp_order):
    '''
    Performs the resampling
    :param data_array: Data array to resample
    :param pixdim_init: Initial pixel dimension
    :param pixdim_fin: Targeted pixel dimension
    :param interp_order: Interpolation order applied
    :return data_resampled: Array containing the resampled data
    '''
    if data_array is None:
        # do nothing
        return
    if np.array_equal(pixdim_fin, pixdim_init):
        return data_array
    to_multiply = np.divide(pixdim_init, pixdim_fin[:len(pixdim_init)])
    data_shape = data_array.shape
    if len(data_shape) != 5:
        raise ValueError("only supports 5D array resampling, "
                         "input shape {}".format(data_shape))
    data_resampled = []
    for t in range(0, data_shape[3]):
        data_mod = []
        for m in range(0, data_shape[4]):
            data_3d = data_array[..., t, m]
            data_new = scipy.ndimage.zoom(data_3d,
                                          to_multiply[0:3],
                                          order=interp_order)
            data_mod.append(data_new[..., np.newaxis, np.newaxis])
        data_mod = np.concatenate(data_mod, axis=-1)
        data_resampled.append(data_mod)
    data_resampled = np.concatenate(data_resampled, axis=-2)
    return data_resampled


### end of resample/reorientation original volumes


def split_filename(file_name):
    '''
    Operation on filename to separate path, basename and extension of a filename
    :param file_name: Filename to treat
    :return pth, fname, ext:
    '''
    pth = os.path.dirname(file_name)
    fname = os.path.basename(file_name)

    ext = None
    for special_ext in FILE_EXTENSIONS:
        ext_len = len(special_ext)
        if fname[-ext_len:].lower() == special_ext:
            ext = fname[-ext_len:]
            fname = fname[:-ext_len] if len(fname) > ext_len else ''
            break
    if ext is None:
        fname, ext = os.path.splitext(fname)
    return pth, fname, ext


def csv_cell_to_volume_5d(csv_cell):
    """
    This function create a 5D image matrix from a csv_cell
    :param csv_cell: an array of file names, e.g. ['path_to_T1', 'path_to_T2']
    :return: 5D image consisting of images from 'path_to_T1', 'path_to_T2'
             The five dimensions are H x W x D x Modality x Time point
    """
    if csv_cell is None:
        return None
    numb_tp = csv_cell.num_time_point
    numb_mod = csv_cell.num_modality
    max_time = numb_tp
    max_mod = numb_mod

    expand_modality_dim = True if numb_mod == 1 else False
    expand_time_point_dim = True if numb_tp == 1 else False

    flag_dimensions_set = False
    dimensions = []
    data_array = []
    for t in range(0, numb_tp):
        data_array.append([])
        for m in range(0, numb_mod):
            data_array[t].append([])
            if not os.path.exists(csv_cell()[t][m]):
                data_array[t][m] = expand_to_5d(np.zeros(dimensions))
                continue
            # load a 3d volume
            img_nii = load_image(csv_cell()[t][m])
            img_data_shape = img_nii.header.get_data_shape()
            assert np.prod(img_data_shape) > 1

            if not flag_dimensions_set:
                dimensions = img_data_shape[0:min(3, len(img_data_shape))]
                flag_dimensions_set = True
            else:
                if not np.all(img_data_shape[:3] == dimensions[:3]):
                    raise ValueError("The 3d dimensionality of image %s "
                                     "%s is not consistent with %s "
                                     % (csv_cell()[m][t],
                                        ' '.join(map(str, img_data_shape[0:3])),
                                        ' '.join(map(str, dimensions))))
            if len(img_data_shape) >= 4 and img_data_shape[3] > 1 \
                    and not expand_time_point_dim:
                raise ValueError("You cannot provide already stacked time "
                                 "points if you stack additional time points ")
            elif expand_time_point_dim and len(img_data_shape) >= 4:
                max_time = max(img_data_shape[3], max_time)
            if len(img_data_shape) >= 5 and img_data_shape[4] > 1 \
                    and not expand_modality_dim:
                raise ValueError("You cannot provide already stacked "
                                 "modalities "
                                 " if you stack additional modalities ")
            elif expand_modality_dim and len(img_data_shape) == 5:
                max_mod = max(max_mod, img_data_shape[4])
            img_data = img_nii.get_data().astype(np.float32)
            img_data = expand_to_5d(img_data)
            img_data = np.swapaxes(img_data, 4, 3)
            data_array[t][m] = img_data
    if expand_modality_dim or expand_time_point_dim:
        data_array = pad_zeros_to_5d(data_array, max_mod, max_time)
    data_to_save = create_5d_from_array(data_array)
    return data_to_save


def expand_to_5d(img_data):
    '''
    Expands an array up to 5d if it is not the case yet
    :param img_data:
    :return:
    '''
    while img_data.ndim < 5:
        img_data = np.expand_dims(img_data, axis=-1)
    return img_data


def pad_zeros_to_5d(data_array, max_mod, max_time):
    '''
    Performs padding of element of a data array if not all modalities or time
    points are present in the data cells.
    :param data_array: data_array (1st dimension time, 2nd modalities)
    :param max_mod: number of modalities
    :param max_time: number of time points to consider
    :return data_array: Data_array with consistent data cells with the number
    of modalities and number of time points
    '''
    if len(data_array) == max_time and len(data_array[0]) == max_mod:
        return data_array
    if len(data_array) == 1:  # Time points already agregated in the
        # individual data array cells
        for m in range(0, len(data_array[0])):
            if data_array[0][m].shape[4] < max_time:
                data = data_array[0][m]
                zeros_to_append = np.zeros([data.shape[0],
                                            data.shape[1],
                                            data.shape[2],
                                            data.shape[3],
                                            max_time - data.shape[4]])
                data_array[0][m] = np.concatenate(
                    [data, zeros_to_append], axis=4)
    else:
        for t in range(0, len(data_array)):
            data = data_array[t][0]
            if data.shape[3] < max_mod:
                zeros_to_append = np.zeros([data.shape[0],
                                            data.shape[1],
                                            data.shape[2],
                                            max_mod - data.shape[3],
                                            data.shape[4]])
                data_array[t][0] = np.concatenate(
                    [data, zeros_to_append], axis=3)
    return data_array


def create_5d_from_array(data_array):
    '''
    From a array of separate data elements, create the final 5d array to use.
     The first dimension of the data_array is time, the second is modalities
    :param data_array: array of sub data elements to concatenate into a 5d
    element
    :return data_5d: Resulting 5d array (3 spatial dimension, modalities, time)
    '''
    data_5d = []
    for t in range(0, len(data_array)):
        data_mod_temp = []
        for m in range(0, len(data_array[0])):
            data_temp = data_array[t][m]
            data_mod_temp.append(data_temp)
        data_mod_temp = np.concatenate(data_mod_temp, axis=3)
        data_5d.append(data_mod_temp)
    data_5d = np.concatenate(data_5d, axis=4)
    return data_5d


def save_volume_5d(img_data, filename, save_path, affine=np.eye(4)):
    '''
    Save the img_data to nifti image
    :param img_data: 5d img to save
    :param filename: filename under which to save the img_data
    :param save_path:
    :param affine: an affine matrix.
    :return:
    '''
    if img_data is None:
        return
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    except OSError:
        tf.logging.fatal('writing output images failed.')
        raise

    img_nii = nib.Nifti1Image(img_data, affine)
    #img_nii.set_data_dtype(np.dtype(np.float32))
    output_name = os.path.join(save_path, filename)
    try:
        nib.save(img_nii, output_name)
    except OSError:
        tf.logging.fatal("writing failed {}".format(output_name))
        raise
    print('Saved {}'.format(output_name))


def match_volume_shape_to_patch_definition(image_data, patch):
    '''
    Adjusts the shape of the image data to match the requested
    patch. This depends on the patch.spatial_rank.
    For spatial rank 2.5 and 3, reshapes to 4D input volume [H x W x D x Modalities]
    For spatial rank 2, reshapes to 4D input volume [H x W x 1 x Modalities]
    '''
    if image_data is None:
        return None
    if patch is None:
        return None
    # spatial_shape = list(image_data.shape[:int(np.ceil(patch.spatial_rank))])
    # spatial_shape += [1]*(3-int(np.ceil(patch.spatial_rank)))
    # return np.reshape(image_data,spatial_shape+[-1])

    #  always casting to 4D input volume [H x W x D x Modality]
    while image_data.ndim > 4:
        image_data = image_data[..., 0]
    while image_data.ndim < 4:
        image_data = np.expand_dims(image_data, axis=-1)
    return image_data


def spatial_padding_to_indexes(spatial_padding):
    '''
    Utility function to create the indices resulting from padding strategy.
    :param spatial_padding:
    :return indices: list of indices resulting from the padding.
    '''
    indexes = np.zeros((len(spatial_padding), 2), dtype=np.int)
    for (i, s) in enumerate(spatial_padding):
        if len(s) == 1:
            indexes[i] = [s[0], s[0]]
        elif len(s) == 2:
            indexes[i] = [s[0], s[1]]
        else:
            raise ValueError("unknown spatial_padding format")
    return indexes.flatten()


def remove_time_dim(tf_tensor):
    """
    Given a tensorflow tensor, ndims==6 means:
    [batch, x, y, z, time, modality]
    this function remove the time dim if it's one
    """
    if tf_tensor.get_shape().ndims != 6:
        return tf_tensor
    if tf_tensor.get_shape()[4] != 1:
        raise NotImplementedError("time sequences not currently supported")
    return tf.squeeze(tf_tensor, axis=4)
