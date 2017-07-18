# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import warnings

import nibabel as nib
import numpy as np
import scipy.ndimage

image_loaders = [nib.load]
try:
    import niftynet.utilities.simple_itk_as_nibabel
    image_loaders.append(niftynet.utilities.simple_itk_as_nibabel.SimpleITKAsNibabel)
except ImportError:
    warnings.warn('SimpleITK adapter failed to load, reducing the supported file formats.',ImportWarning)

    

warnings.simplefilter("ignore", UserWarning)

FILE_EXTENSIONS = [".nii.gz", ".tar.gz"]


#### utilities for file headers

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
        img=image_loader(filename)
        img = correct_image_if_necessary(img)
        return img
      except nib.filebasedimages.ImageFileError: # if the image_loader cannot handle the type continue to next loader
        pass
    raise nib.filebasedimages.ImageFileError('No loader could load the file') # Throw last error

def correct_image_if_necessary(img):
  # Check that affine matches zooms
  pixdim = img.header.get_zooms()
  if not np.array_equal(np.sqrt(np.sum(np.square(img.affine[0:3, 0:3]), 0)), np.asarray(pixdim)):
    if hasattr(img,'get_sform'): 
      # assume it is a malformed NIfTI and try to fix it
      img=rectify_header_sform_qform(img)
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
    pixdim = img_nii.header.get_zooms()[:3] # TODO: assuming 3 elements
    new_affine = create_affine_pixdim(affine, pixdim)
    img_nii.set_sform(new_affine)
    img_nii.set_qform(new_affine)
    return img_nii



#### end of utilities for file headers


### resample/reorientation original volumes
# Perform the reorientation to ornt_fin of the data array given ornt_init
def do_reorientation(data_array, ornt_init, ornt_fin):
    '''
    Performs the reorientation (changing order of axes)
    :param data_array: Array to reorient
    :param ornt_init: Initial orientation
    :param ornt_fin: Target orientation
    :return data_reoriented: New data array in its reoriented form
    '''
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
    Performs the resampling (used to go to and from anisotropic to isotropic
    data)
    :param data_array: Data array to resample
    :param pixdim_init: Initial pixel dimension
    :param pixdim_fin: Targeted pixel dimension
    :param interp_order: Interpolation order applied
    :return data_resampled: Array containing the resampled data
    '''
    if data_array is None:
        # warnings.warn("None array, nothing to resample")
        return
    if np.array_equal(pixdim_fin, pixdim_init):
        return data_array
    to_multiply = np.divide(pixdim_init[0:], pixdim_fin[0:len(pixdim_init)])
    if len(to_multiply) < data_array.ndim:
        to_multiply = np.pad(to_multiply,
                             (0, data_array.ndim - len(to_multiply)),
                             mode='constant',
                             constant_values=1)
    # resampling each 3d volume in the 5D data
    data_resampled = []
    for t in range(0, data_array.shape[4]):
        data_mod = []
        for m in range(0, data_array.shape[3]):
            data_3d = data_array[..., m, t]
            # interp_order_m = interp_order[min(len(interp_order) - 1, m)]
            data_new = scipy.ndimage.zoom(data_3d,
                                          to_multiply[0:3],
                                          order=interp_order)
            data_mod.append(data_new[..., np.newaxis])
        data_mod = np.concatenate(data_mod, axis=-1)
        data_resampled.append(data_mod[..., np.newaxis])
    data_resampled = np.concatenate(data_resampled, axis=-1)
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


def save_volume_5d(img_data, filename, save_path, img_ref=None):
    '''
    Save the img_data to nifti image
    :param img_data: 5d img to save
    :param filename: filename under which to save the img_data
    :param save_path:
    :param img_ref: reference img to use for the setting of header.
    :return:
    '''
    if img_data is None:
        return
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_ref = correct_image_if_necessary(img_ref)
    affine = img_ref.affine
    img_nii = nib.Nifti1Image(img_data, affine)
    img_nii.set_data_dtype(np.dtype(np.float32))
    output_name = os.path.join(save_path, filename)
    nib.save(img_nii, output_name)
    print('Saved {}'.format(output_name))


def match_volume_shape_to_patch_definition(image_data, patch):
    '''
    Reduce 5d to 4d (time series not allowed so far) or augment to 4d
    :param image_data:
    :param patch:
    :return:
    '''
    if image_data is None:
        return None
    if patch is None:
        return None
    # always casting to 4D input volume [H x W x D x Modality]
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


    # def adapt_to_shape(img_to_change, shape, mod='tile'):
    #     if img_to_change is None or img_to_change.size == 0:
    #         return np.zeros(shape)
    #     shape_to_change = img_to_change.shape
    #     if len(shape) < len(shape_to_change):
    #         raise ValueError('shape inconsistency')
    #     if np.all(shape_to_change == shape):
    #         return img_to_change
    #     new_img = np.resize(img_to_change, shape)
    #     return new_img


    # # Check compatibility in dimensions for the first 3 dimensions of two images
    # def check_shape_compatibility_3d(img1, img2):
    #     # consider by default that there are a min of 3 dimensions (2d images are
    #     # always expanded beforehand
    #     if img1.ndim < 3 or img2.ndim < 3:
    #         raise ValueError
    #     return np.all(img1.shape[:3] == img2.shape[:3])

    # def create_new_filename(filename_init, new_path='', new_prefix='',
    #                         new_suffix=''):
    #     path, name, ext = split_filename(filename_init)
    #     if new_path is None or len(new_path) == 0:
    #         new_path = path
    #     new_name = "%s_%s_%s" % (new_prefix, name, new_suffix)
    #     new_filename = os.path.join(new_path, new_name + ext)
    #     new_filename = clean_name(new_filename)
    #     return new_filename


    # def clean_name(filename):
    #     filename = filename.replace("__", "_")
    #     filename = filename.replace("..", ".")
    #     filename = filename.replace("_.", ".")
    #     return filename


    # def load_volume(filename,
    #                 allow_multimod_single_file=False,
    #                 allow_timeseries=False):
    #     if not os.path.exists(filename):
    #         warnings.warn("This file %s does not exist" % filename)
    #         return None
    #
    #     print(filename)
    #     img_nii = nib.load(filename)
    #     img_shape = img_nii.header.get_data_shape()
    #     img_data = img_nii.get_data().astype(np.float32)
    #     if len(img_shape) == 2:  # If the image is 2D it is expanded as a 3D
    #         return np.expand_dims(img_data, axis=2)
    #
    #     if len(img_shape) == 3:  # do nothing if image is 3D
    #         return img_data
    #
    #     if len(img_shape) == 4:  # 4D depends on use of multi time series
    #         warnings.warn("A 4D image has been detected. As per Nifti "
    #                       "standards, it will be considered as a time series "
    #                       "image")
    #         if not allow_timeseries:  # if no time series allowed, take only
    #             # the first volume
    #             warnings.warn("Time series not allowed in this setting, "
    #                           "only the first volume will be returned")
    #             return img_data[..., 0:1]
    #         else:
    #             # time series are moved to the 5th dimension
    #             return np.swapaxes(np.expand_dims(img_data, axis=4), 4, 3)
    #
    #     if len(img_shape) == 5:
    #         warnings.warn("A 5D image has been detected. As per Nifti "
    #                       "conventions, it will be considered as a multimodal image")
    #         if not allow_multimod_single_file:
    #             warnings.warn("Multiple modalities in a single file not "
    #                           "allowed in this setting. Only the first "
    #                           "modality will be considered")
    #             if img_shape[3] == 1:  # only one time point in the 4th dimension
    #                 return img_data[..., 0, 0]
    #             else:
    #                 if not allow_timeseries:
    #                     warnings.warn("Time series not allowed in this "
    #                                   "setting, only the first volume of the "
    #                                   "time series will be returned")
    #                     return img_data[..., 0, 0]
    #                 else:
    #                     return np.swapaxes(img_data[..., 0], 4, 3)
    #         else:
    #             if img_shape[3] == 1:  # only one time point in the image series
    #                 return np.swapaxes(img_data[..., 0, :], 4, 3)
    #             elif not allow_timeseries:
    #                 warnings.warn("Time series not allowed in this setting, "
    #                               "only the first volume multimodal will be "
    #                               "returned")
    #                 return np.swapaxes(img_data[..., 0, :], 4, 3)
    #             else:
    #                 return np.swapaxes(img_data, 4, 3)
