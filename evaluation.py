import os.path
import scipy
import nibabel as nib
import numpy as np
from pairwise_measures import PairwiseMeasures
import itertools
import util


MEASURES = (
        'ref volume', 'seg volume',
        'tp', 'fp', 'fn',
        'connected_elements', 'vol_diff',
        'outline_error', 'detection_error',
        'fpr', 'ppv', 'npv', 'sensitivity', 'specificity',
        'accuracy', 'jaccard', 'dice', 'ave_dist', 'haus_dist'
        )
OUTPUT_FORMAT = '{:4f}'

def run(param):
    name_for_file = os.path.split(param.ref_image_dir)[1] + \
            '_' + os.path.split(param.seg_image_dir)[1]
    name_list = None
    if not param.name_ref == '""' :  # file names from a param string
        name_list = param.name_ref
    elif not param.list_file == '""' :  # file names from a text file
        name_list = open("param.list_file", "r").read().split(',')
    else:  # file names by listing the directories
        seg_names = util.list_nifti_files(param.seg_image_dir, param.ext)
        ref_names = util.list_nifti_files(param.ref_image_dir, param.ext)

    if name_list is not None:
        for n in name_list:
            ref_n, seg_n = util.list_img_seg_by_fname(
                    n, param.ref_image_dir, param.seg_image_dir)
            if len(seg_n) != len(ref_n):
                seg_n_list = [item for item in seg_n for i in range(len(ref_n))]
                ref_n_list = [ref_n for i in range(len(seg_n))]
                seg_names.append(seg_n_list)
                ref_names.append(ref_n_list)
            else:
                seg_names.append(seg_n)
                ref_names.append(reg_n)
            name_for_file = name_for_file+'_'.join(str(e) for e in name_list)

    # start writing output
    output_name = os.path.join(
            param.save_eval_dir,
            name_for_file + '_' + param.name_out + '.csv')
    if len(ref_names) != len(seg_names):
        print('Pairwise measurements - multiple ref. vs multiple seg.')
        seg_names_new = [item for item in seg_names \
                for i in range(len(ref_names))]
        seg_names = seg_names_new
        ref_names_new = [ref_names for i in range(len(seg_names))]
        ref_names = list(itertools.chain.from_iterable(ref_names_new))

    print("Writing to {}".format(output_name))
    print("List of references is {}".format(ref_names))
    print("List of segmentations is {}".format(seg_names))

    # calculation and writing
    out_stream = open(output_name, 'w+')
    print >> out_stream, "Name (ref), Name (seg), Label," +\
            PairwiseMeasures(None, None, measures=MEASURES).header_str()
    for i in range(0, len(seg_names)):
        seg_name = seg_names[i]
        ref_name = ref_names[i]
        print('>>> {} of {} evaluations, comparing {} and {}.'.format(
            i + 1, len(seg_names), ref_name, seg_name))
        seg_nii = nib.load(os.path.join(param.seg_image_dir, seg_name))
        ref_nii = nib.load(os.path.join(param.ref_image_dir, ref_name))
        PixDim = seg_nii.header.get_zooms()[0:3]
        seg = seg_nii.get_data()
        ref = ref_nii.get_data()
        assert(np.all(seg) >= 0)
        assert(np.all(ref) >= 0)
        assert(seg.shape == ref.shape)

        if (param.seg_type == 'discrete') and (np.max(seg) <= 1)\
                and (len(np.unique(seg)) > 2):
            print('Not integer class labels for discrete analysis')
            print('Thresholding to binary map with threshold: {}'.format(
                param.threshold))
            seg = np.asarray(seg >= param.threshold, dtype=np.int8)

        if param.seg_type == 'discrete':
            print('Discrete analysis')
            threshold_steps = np.unique(ref)
        elif param.seg_type == 'prob':
            print('Probabilistic analysis')
            threshold_steps = np.arange(0, 1, param.step)

        for i in threshold_steps:
            if i==0: continue
            if i >= 1:  # discrete eval
                seg_binary = np.asarray(seg == i, dtype=np.float32)
                ref_binary = np.asarray(ref == i, dtype=np.float32)
            elif i < 1:  # prob eval
                seg_binary = np.asarray(seg >= i, dtype=np.float32)
                ref_binary = np.asarray(ref >= 0.5, dtype=np.float32)
            if np.all(seg_binary==0):
                print("Empty foreground in thresholded binary image.")
                continue
            PE = PairwiseMeasures(seg_binary, ref_binary,
                    measures=MEASURES, num_neighbors=6, pixdim=PixDim)
            fixed_fields = "{}, {}, {},".format(ref_name, seg_name, i)
            print >> out_stream, fixed_fields + PE.to_string(OUTPUT_FORMAT)
