# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import nibabel
import tensorflow as tf

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

def load_file(img_name, seg_name=None):
    img_data = nibabel.load(img_name).get_data()
    if img_data.ndim == 4:
        img_data = img_data[:, :, :, 0]
    seg_data = nibabel.load(seg_name).get_data().astype(np.int64)\
        if seg_name is not None else None
    return img_data, seg_data


def list_nifti_files(img_dir, rand=False):
    # TODO check images and labels consistency
    train_names = [fn for fn in os.listdir(img_dir)
                   if fn.endswith(".nii.gz")]
    if train_names == []:
        print 'no files in {}'.format(img_dir)
        raise IOError
    if rand:
        random.shuffle(train_names)
    return train_names

def has_bad_inputs(args):
    print 'Input params:'
    for arg in vars(args):
        user_value = getattr(args, arg)
        if user_value is None:
            print '{} not set'.format(arg)
            return True
        print "-- {}: {}".format(arg, getattr(args, arg))

    # at each iteration [batch_size] samples will be read from queue
    if args.queue_length < args.batch_size:
        print 'queue_length ({}) should be >= batch_size ({}).'.format(
                args.queue_length, args.batch_size)
        return True
    return False


def volume_of_zeros_like(image_name, dtype=np.int64):
    # initialise a 3D volume of zeros, with the same shape as image_names
    ori_img = nibabel.load(image_name).get_data()
    ori_img = ori_img[:, :, :, 0] \
        if ori_img.ndim == 4 else ori_img
    new_volume = np.zeros_like(ori_img, dtype=np.int64)
    return new_volume


def save_segmentation(param, img_name, pred_img):
    if img_name is None:
        return
    if pred_img is None:
        return
    # TODO warning if save to label_dir
    pred_folder = "{}_pred_{}/".format(param.save_seg_dir, param.pred_iter)
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)
    save_name = pred_folder + img_name

    # TODO  randomise names to avoid overwrite
    #import random
    #if os.path.isfile(save_name): # prediction file exist
    #  save_name = save_name[:-7] + time.strftime("%Y%m%d-%H%M%S")
    #  save_name = save_name + random.choice('abcdefghinm') + '.nii.gz'
    #pred_img = (label_map[pred_img.astype(np.int64)]).astype(np.int64)
    (w, h, d) = pred_img.shape
    if param.volume_padding_size > 0: # remove paddings
        pred_img = pred_img[
                param.volume_padding_size : (w - param.volume_padding_size),
                param.volume_padding_size : (h - param.volume_padding_size),
                param.volume_padding_size : (d - param.volume_padding_size)]
    ori_aff = nibabel.load(param.eval_image_dir + '/' + img_name).affine
    predicted_nii = nibabel.Nifti1Image(pred_img, ori_aff)
    predicted_nii.set_data_dtype(np.dtype(np.float32))
    nibabel.save(predicted_nii, save_name)
    print 'saved %s' % save_name
