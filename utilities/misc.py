# -*- coding: utf-8 -*-
import os
import random
from functools import partial

from skimage import measure
from scipy import ndimage
import nibabel
import numpy as np
import tensorflow as tf

LABEL_STRINGS = ['Label', 'LABEL', 'label']


def average_grads(tower_grads):
    # average gradients computed from multiple GPUs
    ave_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        ave_grads.append(grad_and_var)
    return ave_grads


def _guess_fullname(data_dir, patient_id, modality):
    if not isinstance(modality, list):
        modality = [modality]
    names = [os.path.join(data_dir, '%s_%s' % (patient_id, m))
             for m in modality]
    for n in names:
        if os.path.exists(n + '.nii'):
            return n + '.nii'
        elif os.path.exists(n + '.nii.gz'):
            return n + '.nii.gz'
    return None


def _split_fullname(name):
    lower_name = name.lower()
    if lower_name.endswith((".nii")):
        return [name[:-4], '.nii']
    elif lower_name.endswith((".nii.gz")):
        return [name[:-7], '.nii.gz']
    else:
        return None


def _guess_img_affine(data_dir, patient_id):
    # read volume affine from data folder for the patient_id
    first_mod = list_modality(data_dir)[0]
    file_name = _guess_fullname(data_dir, patient_id, first_mod)
    return nibabel.load(file_name).affine


def list_patient(data_dir):
    id_list = []
    for file_name in os.listdir(data_dir):
        patient_id = _split_fullname(file_name)
        if patient_id is None:
            continue
        id_list.append(patient_id[0].split('_')[0])
    return np.unique(id_list).tolist()


def list_modality(data_dir):
    # file name format: 'patient_modality.extension'
    mod_list = []
    for file_name in os.listdir(data_dir):
        file_name = _split_fullname(file_name)
        if file_name is None:
            continue
        modality = file_name[0].split('_')[-1]
        if modality not in LABEL_STRINGS:
            mod_list.append(modality)
    return np.unique(mod_list).tolist()


def load_file(data_dir, patient_id, with_seg=False):
    # file name format is assumed to be 'patient_modality.extension'
    # load image data with shape [d_z, d_y, d_x, d_mod]
    img_data = []
    for mod in list_modality(data_dir):
        img_name = _guess_fullname(data_dir, patient_id, mod)
        if img_name is None:
            raise ValueError('Not found: %s/%s_%s.[nii|nii.gz]' % (
                data_dir, patient_id, mod))
        img = nibabel.load(img_name).get_data().astype(np.float32)
        img = img[:, :, :, 0] if img.ndim == 4 else img
        img_data.append(img)
    img_data = np.stack(img_data, axis=-1)

    # load segmentation data with shape [d_z, d_y, d_x] if exists
    if not with_seg:
        return img_data, None
    seg_name = _guess_fullname(data_dir, patient_id, LABEL_STRINGS)
    seg_data = nibabel.load(seg_name).get_data().astype(np.int64)\
        if seg_name is not None else None
    return img_data, seg_data

def list_files(img_dir, ext='.nii.gz'):
    names = [fn for fn in os.listdir(img_dir) if fn.lower().endswith(ext)]
    if not names:
        print('no files in {}'.format(img_dir))
        raise IOError
    return names

def has_bad_inputs(args):
    print('Input params:')
    for arg in vars(args):
        user_value = getattr(args, arg)
        print("-- {}: {}".format(arg, getattr(args, arg)))
        if user_value is None:
            print('{} not set in the config file'.format(arg))
            return True
    ## at each iteration [batch_size] samples will be read from queue
    #if args.queue_length < args.batch_size:
    #    print('queue_length ({}) should be >= batch_size ({}).'.format(
    #        args.queue_length, args.batch_size))
    #    return True
    return False

def volume_of_zeros_like(data_dir, patient_name, mod_name, d_type=np.int64):
    # initialise a 3D volume of zeros, with the same shape as image_names
    img_name = _guess_fullname(data_dir, patient_name, mod_name)
    ori_img = nibabel.load(img_name).get_data()
    ori_img = ori_img[:, :, :, 0] if ori_img.ndim == 4 else ori_img
    new_volume = np.zeros_like(ori_img, dtype=d_type)
    return new_volume

def save_segmentation(param, pat_name, pred_img):
    if pat_name is None:
        return
    if pred_img is None:
        return
    # TODO warning if save to label_dir
    pred_folder = "{}_iter_{}/".format(param.save_seg_dir, param.pred_iter)
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)
    save_name = os.path.join(pred_folder, '%s%s' % (pat_name, '.nii.gz'))

    # TODO  randomise names to avoid overwrite
    # import random
    # if os.path.isfile(save_name): # prediction file exist
    #  save_name = save_name[:-7] + time.strftime("%Y%m%d-%H%M%S")
    #  save_name = save_name + random.choice('abcdefghinm') + '.nii.gz'
    # pred_img = (label_map[pred_img.astype(np.int64)]).astype(np.int64)
    (w, h, d) = pred_img.shape
    if param.volume_padding_size > 0:  # remove paddings
        pred_img = pred_img[
                   param.volume_padding_size: (w - param.volume_padding_size),
                   param.volume_padding_size: (h - param.volume_padding_size),
                   param.volume_padding_size: (d - param.volume_padding_size)]

    ori_aff = _guess_img_affine(param.eval_data_dir, pat_name)
    predicted_nii = nibabel.Nifti1Image(pred_img, ori_aff)
    predicted_nii.set_data_dtype(np.dtype(np.float32))
    nibabel.save(predicted_nii, save_name)
    print('saved %s' % save_name)


class MorphologyOps(object):
    def __init__(self, binary_img, neigh):
        self.binary_map = np.asarray(binary_img, dtype=np.int8)
        self.neigh = neigh

    def border_map(self):
        west = ndimage.shift(self.binary_map, [-1, 0, 0], order=0)
        east = ndimage.shift(self.binary_map, [1, 0, 0], order=0)
        north = ndimage.shift(self.binary_map, [0, 1, 0], order=0)
        south = ndimage.shift(self.binary_map, [0, -1, 0], order=0)
        top = ndimage.shift(self.binary_map, [0, 0, 1], order=0)
        bottom = ndimage.shift(self.binary_map, [0, 0, -1], order=0)
        cumulative = west + east + north + south + top + bottom
        border = ((cumulative < 6) * self.binary_map) == 1
        return border

    def foreground_component(self):
        return measure.label(self.binary_map, background=0)


class CacheFunctionOutput(object):
    """
    this provides a decorator to cache function outputs
    to avoid repeating some heavy function computations
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, _=None):
        if obj is None:
            return self
        return partial(self, obj)  # to remember func as self.func

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            value = cache[key]
        except KeyError:
            value = cache[key] = self.func(*args, **kw)
        return value
