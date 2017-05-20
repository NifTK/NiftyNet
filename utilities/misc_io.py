import os
import warnings

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


#### utilities for file headers

def create_affine_pixdim(affine, pixdim):
    norm_affine = np.sqrt(np.sum(np.square(affine[:, 0:3]), 0))
    to_divide = np.tile(
        np.expand_dims(np.append(norm_affine, 1), axis=1), [1, 4])
    to_multiply = np.tile(
        np.expand_dims(np.append(np.asarray(pixdim), 1), axis=1), [1, 4])
    affine_fin = np.multiply(np.divide(affine, to_divide.T), to_multiply.T)
    return affine_fin


def rectify_header_sform_qform(img_nii):
    # TODO: check img_nii is a nibabel object
    pixdim = img_nii.header.get_zooms()
    sform = img_nii.get_sform()
    qform = img_nii.get_qform()
    norm_sform = np.sqrt(np.sum(np.square(sform[0:3, 0:3]), 0))
    norm_qform = np.sqrt(np.sum(np.square(qform[0:3, 0:3]), 0))
    flag_sform_problem = False
    flag_qform_problem = False
    if not np.array_equal(norm_sform, np.asarray(pixdim)):
        warnings.warn("Incompatibility between header pixdim and sform")
        flag_sform_problem = True
    if not np.array_equal(norm_qform, np.asarray(pixdim)):
        warnings.warn("Incompatibility between header pixdim and qform")
        flag_qform_problem = True

    if not flag_qform_problem and not flag_sform_problem:
        return img_nii

    if flag_sform_problem and img_nii.get_header()['sform_code'] > 0:
        if not flag_qform_problem:
            img_nii.set_sform(np.copy(img_nii.get_qform()))
            return img_nii
        else:
            affine = img_nii.affine
            pixdim = img_nii.header.get_zooms()
            new_affine = create_affine_pixdim(affine, pixdim)
            img_nii.set_sform(new_affine)
            img_nii.set_qform(new_affine)
            return img_nii
    else:
        img_nii.set_qform(np.copy(img_nii.get_sform()))
        return img_nii


#### end of utilities for file headers

### resample/reorientation original volumes

# Perform the reorientation to ornt_fin of the data array given ornt_init
def do_reorientation(data_array, ornt_init, ornt_fin):
    if np.array_equal(ornt_init, ornt_fin):
        #print("Already in same orientation")
        return data_array
    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)
    data_reoriented = nib.orientations.apply_orientation(
        data_array, ornt_transf)
    return data_reoriented


# Perform the resampling of the data array given the initial and final pixel
# dimensions and the interpolation order
def do_resampling(data_array, pixdim_init, pixdim_fin, interp_order=3):
    if np.array_equal(pixdim_fin, pixdim_init):
        #print("Already with similar resolution")
        return data_array
    to_multiply = np.divide(pixdim_init, pixdim_fin)
    if len(to_multiply) < data_array.ndim:
        to_multiply = np.pad(to_multiply,
                             (0, data_array.ndim - len(to_multiply)),
                             mode='constant', constant_values=1)
    data_resampled = zoom(data_array, to_multiply, order=interp_order)
    return data_resampled


### end of resample/reorientation original volumes

# Load volume expected and return 3D and not 4D
def load_volume(filename, allow_multimod_single_file=False, allow_timeseries=False):
    if not os.path.exists(filename):
        warnings.warn("This file %s does not exist" % filename)
        return None
    else:
        print filename
        img_nii = nib.load(filename)
        img_shape = img_nii.header.get_data_shape()
        img_data = img_nii.get_data().astype(np.float32)
        if len(img_shape) == 2:  # If the image is 2D it is expanded as a 3D
            return np.expand_dims(img_data, axis=2)
        if len(img_shape) == 3:  # Nothing to do if image is 3D
            return img_data
        if len(img_shape) == 4:  # Case img is 4D depends on use of multi
            # time series
            warnings.warn("A 4D image has been detected. As per Nifti "
                          "standards, it will be considered as a time series "
                          "image")
            if not allow_timeseries:  # if no time series allowed, take only
                # the first volume
                warnings.warn("Time series not allowed in this setting, "
                              "only the first volume will be returned")
                return img_data[..., 0:1]
            else:  # In the nifty net handling, the time series are moved to
                # the 5th dimension
                return np.swapaxes(np.expand_dims(img_data, axis=4), 4, 3)
        if len(img_shape) == 5:
            warnings.warn("A 5D image has been detected. As per Nifti "
                          "conventions, it will be considered as a multimodal image")
            if not allow_multimod_single_file:
                warnings.warn("Multiple modalities in a single file not "
                              "allowed in this setting. Only the first "
                              "modality will be considered")
                if img_shape[3] == 1:  # Case where there is only one time
                    # point in the 4th dimension
                    return img_data[..., 0, 0]
                else:
                    if not allow_timeseries:
                        warnings.warn("Time series not allowed in this "
                                      "setting, only the first volume of the "
                                      "time series will be returned")
                        return img_data[..., 0, 0]
                    else:
                        return np.swapaxes(img_data[..., 0], 4, 3)
            else:
                if img_shape[3] == 1:  # only one time point in the image
                    # series
                    return np.swapaxes(img_data[..., 0, :], 4, 3)
                elif not allow_timeseries:
                    warnings.warn("Time series not allowed in this setting, "
                                  "only the first volume multimodal will be "
                                  "returned")
                    return np.swapaxes(img_data[..., 0, :], 4, 3)
                else:
                    return np.swapaxes(img_data, 4, 3)



def split_filename(file_name):

    special_extensions = [".nii.gz", ".tar.gz"]

    pth = os.path.dirname(file_name)
    fname = os.path.basename(file_name)

    ext = None
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and \
                (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = os.path.splitext(fname)

    return pth, fname, ext
