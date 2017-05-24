
import nibabel as nib
import warnings
import numpy as np
import numpy.ma as ma
import utilities.misc as util
import utilities.misc_io as io
import os
import scipy.ndimage as ndimg
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
import utilities.constraints_classes as cc
import utilities.misc_csv as misc_csv
from scipy.signal import argrelextrema
from shutil import copyfile
from skimage import data
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters



def percentiles(img, mask, perc):
    masked_img = ma.masked_array(img, mask)
    perc_results = np.percentile(ma.compressed(masked_img),
                                 100*np.array(perc))
    hist, bin = np.histogram(ma.compressed(masked_img), bins=50)
    return perc_results


def standardise_cutoff(cutoff, type_hist='quartile'):
    if len(cutoff) == 0:
        return [0.01, 0.99]
    if len(cutoff) > 2:
        cutoff = np.unique([np.min(cutoff), np.max(cutoff)])
    if len(cutoff) == 1:
        if cutoff[0] < 0.5:
            cutoff[1] = 1.
        else:
            cutoff[1] = cutoff[0]
            cutoff[0] = 0.
    if cutoff[0] > cutoff[1]:
        temp = cutoff[0]
        cutoff[0] = cutoff[1]
        cutoff[1] = temp
    if cutoff[0] < 0.:
        cutoff[0] = 0.
    if cutoff[1] > 1.:
        cutoff[1] = 1.
    # if type_hist == 'percentile':
    #     cutoff[0] = np.min([cutoff[0], 0.1])
    #     cutoff[1] = np.max([cutoff[1], 0.9])
    # if type_hist == 'quartile':
    #     cutoff[0] = np.min([cutoff[0], 0.25])
    #     cutoff[1] = np.max([cutoff[1], 0.75])
    return cutoff

def create_database_perc_multimod_arrayfiles(mask_type,
                                             array_files,
                                             cutoff,
                                             list_modalities=None):
    perc_database = {}
    for p in array_files:
    #for i in range(0, 10):
        #p = array_files[i]
        img_data = io.prepare_5d_data(p)
        numb_modalities = img_data.shape[3]
        numb_timepoints = img_data.shape[4]
        if list_modalities is None:
            to_do = {}
            for i in range(0, numb_modalities):
                to_do['Mod'+i] = i
        else:
            to_do = {m: list_modalities[m] for m in list_modalities.keys() if
                     list_modalities[m] < numb_modalities}

        for m in to_do.keys():
            if m not in perc_database.keys():
                perc_database[m] = []
            for t in range(0, numb_timepoints):
                img_temp = img_data[..., to_do[m], t]
                mask_temp = create_mask_img_3d(img_temp, mask_type)
                perc = percentiles(img_temp, 1 - mask_temp, [cutoff[0], 0.1, 0.2,
                                                             0.25, 0.3, 0.4, 0.5,
                                                             0.6, 0.7,
                                                             0.75, 0.8, 0.9,
                                                             cutoff[1]])
                if len(perc_database[m]) == 0:
                    perc_database[m] = perc
                else:
                    perc_database[m] = np.vstack([perc_database[m], perc])
    return perc_database



# Create the database of landmarks using multimodal (5D) nifti images
def create_database_perc_list_multimod(mask_type, list_files, cutoff,
                                       type_hist='quartile'):
    perc_database = []
    for file in list_files:
        img_data = io.load_volume(file, allow_multimod=True,
                                  allow_timeseries=True)
        if img_data.ndim == 3:
            img_data = np.expand_dims(np.expand_dims(img_data, axis=3), axis=4)
        if img_data.ndim == 4:
            img_data = np.expand_dims(np.expand_dims(img_data), axis=4)
        numb_modalities = img_data.shape[3]
        numb_timepoints = img_data.shape[4]

        for m in range(0, numb_modalities):
            perc_database.append([])
            for t in range(0, numb_timepoints):
                img_temp = img_data[..., m, t]
                mask_temp = create_mask_img_3d(img_temp, mask_type)
                if np.count_nonzero(img_temp) > 0:
                    perc = percentiles(img_temp, 1-mask_temp, [cutoff[0], 0.1, 0.2,
                                                      0.25, 0.3, 0.4, 0.5, 0.6, 0.7,
                            0.75, 0.8, 0.9, cutoff[1]])
                    if len(perc_database[m]) == 0:
                        perc_database[m] = perc
                    else:
                        perc_database[m] = np.vstack(perc_database[m],perc)





# Create the database of landmarks to build the standardisation model using
# the directories
def create_database_perc_dir(mask_type, list_paths, modality, cutoff,
                             type_hist='quartile'):

    perc_database = None
    list_patients = util.list_subjects(list_paths,
                                       modality)
    cutoff = standardise_cutoff(cutoff, type_hist)
    for p in list_patients:
    #for i in range(0, 100):
        #p = list_patients[i]
        file_img = io.create_list_modalities_available(list_paths, p)
        img_nii = nib.load(file_img[modality][0])
        img = img_nii.get_data()
        mask = create_mask_img_3d(img, mask_type)
        mask[mask > 0.5] = 1
        perc = percentiles(img, 1 - mask,
                           [cutoff[0], 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7,
                            0.75, 0.8, 0.9, cutoff[1]])
        if perc_database is None:
            perc_database = perc
        else:
            perc_database = np.vstack((perc_database, perc))
    return perc_database


def create_database_perc(img_names, mask_type, cutoff, type_hist='quartile'):
    perc_database = None
    cutoff = standardise_cutoff(cutoff, type_hist)
    for name in img_names:
        img_nii = nib.load(name)

        img = img_nii.get_data().astype(np.float32)
        if np.count_nonzero(img) > 0:
            mask = create_mask_img_3d(img,mask_type)
            mask[mask > 0.5] = 1
            perc = percentiles(img, 1-mask,
                           [cutoff[0], 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7,
                            0.75, 0.8, 0.9, cutoff[1]])
            if perc_database == None:
                perc_database = perc
            else:
                perc_database = np.vstack((perc_database, perc))
    return perc_database


def create_standard_range(perc_database):
    # if pc1 > pc2:
    #     temp = pc2
    #     pc2 = pc1
    #     pc1 = temp
    # if pc1 < 0:
    #     pc1 =0
    # if pc2 > 16:
    #     pc2 = 16
    # if type == 'quartile':
    #     pc1 = np.min([pc1, 5])
    #     pc2 = np.max([pc2, 11])
    # if type == 'percentile':
    #     pc1 = np.min([pc1, 3])
    #     pc2 = np.max([pc2, 13])
    left_side = perc_database[:, 6] - perc_database[:, 0]
    right_side = perc_database[:, 12] - perc_database[:, 6]
    range_min = (np.max(left_side)+np.max(right_side))*np.max\
        ([np.max(left_side)/np.min(left_side),
          np.max(right_side)/np.min(right_side)])
    return 0., 100.


def read_mapping_file(file_mapping, modality='T1'):
    with open(file_mapping, "r") as f:
        for line in f:
            if len(line) > 2:
                if line.split()[0] == modality:
                    return map(float, line.split()[1:])


def write_mapping_file(mapping, filename, modality='T1'):
    if os.path.exists(filename):
        f = open(filename, 'a')
    else:
        f = open(filename, 'w+')
    f.write("\n")
    string_mapping = ' '.join(map(str, mapping))
    string_fin = ('%s %s') % (modality, string_mapping)
    f.write(string_fin)
    # f.write(modality)
    # for i in mapping:
    #     f.write(" %s" % map(str,i))
    #f.write("\n")

def write_mapping_file_multimod(mapping_multimod, filename):
    for m in mapping_multimod.keys():
        write_mapping_file(mapping_multimod[m], filename, modality=m)



# create mask for image if multimodal or not
def create_mask_img_multimod(img, type_mask='otsu_plus', alpha=0.1,
                             multimod=[0], multimod_type='and'):
    if img.ndim == 3:
        thr = alpha*img.mean()
        return create_mask_img_3d(img, type_mask, thr)
    if np.max(multimod) > img.shape[3]:
        raise ValueError
    if len(multimod) == 1:
        thr = alpha*img.mean()

        return create_mask_img_3d(img[..., np.min([multimod[0],
                                  img.shape[3]])], type_mask, thr)
    else:
        mask_init = np.zeros([img.shape[0], img.shape[1], img.shape[2],
                              len(multimod)])
        for i in range(0,len(multimod)):
            thr = alpha*img[..., i].mean()
            mask_temp = create_mask_img_3d(
                img[...,np.min([multimod[i],img.shape[3]])],
                type_mask, thr)
            mask_init[:, :, :, i] = mask_temp

        if multimod_type == 'or':
            # Case when the mask if formed by the union of all modalities masks
            mask_reduced = np.sum(mask_init, axis=3)
            mask_reduced[mask_reduced>0] = 1
            return mask_reduced
        elif multimod_type == 'and':
            # Case when the mask is formed by the intersection of all
            # modalities masks
            mask_reduced = np.sum(mask_init, axis=3)
            mask_reduced[mask_reduced<len(multimod)-1] = 0
            mask_reduced[mask_reduced>0] = 1
            return mask_reduced
        else:
            # Case when there will be one mask for each modality
            return mask_init


def create_mask_img_3d(img, type_mask='otsu_plus', thr=0., fill=True):
    if not img.ndim == 3:
        raise ValueError("At this stage we should only be masking a 3D image")
    mask_init = np.copy(img)
    if type_mask == 'threshold_plus':
        mask_init[img>thr] = 1
        mask_init[img<=thr] = 0
    elif type_mask == 'threshold_minus':
        mask_init[img<thr] = 1
        mask_init[img>=thr] = 0
    elif type_mask == 'otsu_plus':
        if np.count_nonzero(img) == 0:
            thr=0
            warnings.warn("Only zeros in this image...")
        else:
            thr = filters.threshold_otsu(img)
        mask_init[img>thr] = 1
        mask_init[img<=thr] = 0
    elif type_mask == 'otsu_minus':
        thr = filters.threshold_otsu(img)
        mask_init[img<thr] = 1
        mask_init[img>=thr] = 0
    mask_1 = ndimg.binary_dilation(mask_init, iterations=2)
    mask_bis = fill_holes(mask_1)
    #mask_fin = ndimg.binary_erosion(mask_bis, iterations=2)
    return mask_bis

def create_mapping_perc_multimod(perc_database_multi, s1, s2):
    final_map = []
    for m in range(0,len(perc_database_multi)):
        final_map.append(create_mapping_perc(perc_database_multi[m], s1, s2))
    return final_map

def create_mapping_perc(perc_database, s1, s2):
    final_map = np.zeros([perc_database.shape[0], 13])
    for j in range(0, perc_database.shape[0]):
        lin_coeff = (s2 - s1) / (perc_database[j, 12] - perc_database[j, 0])
        affine_coeff = s1 - lin_coeff * perc_database[j, 0]
        for i in range(0, 13):
            final_map[j, i] = lin_coeff * perc_database[j, i] + affine_coeff
    return np.mean(final_map, axis=0)


def transform_for_mapping(img, mask, mapping, cutoff, type_hist='quartile'):
    standardise_cutoff(cutoff, type_hist)
    range_to_use = None
    if type_hist == 'quartile':
        range_to_use = [0, 3, 6, 9, 12]
    if type_hist == 'percentile':
        range_to_use = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]
    if type_hist == 'median':
        range_to_use = [0, 6, 12]
    mask[mask > 0.5] = 1
    perc = percentiles(img, 1-mask, [cutoff[0], 0.1, 0.2, 0.25, 0.3, 0.4, 0.5,
                                      0.6, 0.7, 0.75, 0.8, 0.9, cutoff[1]])
    # Apply linear histogram standardisation
    lin_img = np.ones_like(img, dtype=np.float32)
    aff_img = np.zeros_like(img, dtype=np.float32)
    affine_map = np.zeros([2, len(range_to_use)-1])
    for i in range(len(range_to_use)-1):
        affine_map[0, i] = (mapping[range_to_use[i+1]] - mapping[range_to_use[i]]) / \
                           (perc[range_to_use[i+1]] - perc[range_to_use[i]])
        affine_map[1, i] = mapping[range_to_use[i]] - affine_map[0, i] * perc[range_to_use[i]]
        lin_img[img >= perc[range_to_use[i]]] = affine_map[0, i]
        aff_img[img >= perc[range_to_use[i]]] = affine_map[1, i]
    # Note that values below cutoff[0] over cutoff[1] are also transformed at this stage
    lin_img[img < perc[range_to_use[0]]] = affine_map[0, 0]
    aff_img[img < perc[range_to_use[0]]] = affine_map[1, 0]
    new_img = np.multiply(lin_img, img) + aff_img
    # Apply smooth thresholding (exponential) below cutoff[0] and over cutoff[1]
    low_values = img <= perc[range_to_use[0]]
    new_img[low_values] = smooth_threshold(new_img[low_values], mode='low_value')
    high_values = img >= perc[range_to_use[-1]]
    new_img[high_values] = smooth_threshold(new_img[high_values], mode='high_value')
    # Apply mask and set background to zero
    new_img[mask == 0] = 0.
    return new_img



def smooth_threshold(value, mode='high_value'):
    smoothness = 1.
    if mode == 'high_value':
        affine = np.min(value)
        smooth_value = (value - affine)/smoothness
        smooth_value = (1. - np.exp((-1)*smooth_value)) + affine
    elif mode == 'low_value':
        affine = np.max(value)
        smooth_value = (value - affine)/smoothness
        smooth_value = (np.exp(smooth_value) - 1.) + affine
    else:
        smooth_value = value
    return smooth_value
