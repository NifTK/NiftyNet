from __future__ import absolute_import, print_function

import os.path

import nibabel as nib
import numpy as np

import niftynet.utilities.csv_table as csv_table
from niftynet.evaluation.pairwise_measures import PairwiseMeasures

MEASURES = (
    'ref volume', 'seg volume',
    'tp', 'fp', 'fn',
    'connected_elements', 'vol_diff',
    'outline_error', 'detection_error',
    'fpr', 'ppv', 'npv', 'sensitivity', 'specificity',
    'accuracy', 'jaccard', 'dice', 'ave_dist', 'haus_dist'
)
# MEASURES_NEW = ('ref volume', 'seg volume', 'tp', 'fp', 'fn', 'outline_error',
#             'detection_error', 'dice')
OUTPUT_FORMAT = '{:4f}'
OUTPUT_FILE_PREFIX = 'PairwiseMeasure'


def run(param, csv_dict):
    # output
    out_name = '{}_{}_{}.csv'.format(
        OUTPUT_FILE_PREFIX,
        os.path.split(param.ref_dir)[1],
        os.path.split(param.seg_dir)[1])
    print("Writing {} to {}".format(out_name, param.save_csv_dir))

    # inputs
    csv_loader = csv_table.CSVTable(csv_dict=csv_dict, allow_missing=False)
    seg_names = [csv_loader._csv_table[m][1][0][0] for m in range(
        0, len(csv_loader._csv_table))]
    ref_names = [csv_loader._csv_table[m][2][0][0] for m in range(
        0, len(csv_loader._csv_table))]
    # seg_names = util.list_files(param.seg_dir, param.ext)
    # ref_names = util.list_files(param.ref_dir, param.ext)
    pair_list = list(zip(seg_names, ref_names))
    # TODO check seg_names ref_names matching
    # TODO do we evaluate all combinations?
    # import itertools
    # pair_list = list(itertools.product(seg_names, ref_names))
    print("List of references is {}".format(ref_names))
    print("List of segmentations is {}".format(seg_names))

    # prepare a header for csv
    with open(os.path.join(param.save_csv_dir, out_name), 'w+') as out_stream:
        # a trivial PairwiseMeasures obj to produce header_str
        m_headers = PairwiseMeasures(0, 0, measures=MEASURES).header_str()
        out_stream.write("Name (ref), Name (seg), Label" + m_headers + '\n')

        # do the pairwise evaluations
        for i, pair_ in enumerate(pair_list):
            seg_name = pair_[0]
            ref_name = pair_[1]
            print('>>> {} of {} evaluations, comparing {} and {}.'.format(
                i + 1, len(pair_list), ref_name, seg_name))
            seg_nii = nib.load(os.path.join(param.seg_dir, seg_name))
            ref_nii = nib.load(os.path.join(param.ref_dir, ref_name))
            voxel_sizes = seg_nii.header.get_zooms()[0:3]
            seg = seg_nii.get_data()
            ref = ref_nii.get_data()
            assert (np.all(seg) >= 0)
            assert (np.all(ref) >= 0)
            assert (seg.shape == ref.shape)

            if (param.seg_type == 'discrete') and (np.max(seg) <= 1) \
                    and (len(np.unique(seg)) > 2):
                print('Non-integer class labels for discrete analysis')
                print('Thresholding to binary map with threshold: {}'.format(
                    param.threshold))
                seg = np.asarray(seg >= param.threshold, dtype=np.int8)

            ## TODO: user specifies how to convert seg -> seg_binary
            if param.seg_type == 'discrete':
                print('Discrete analysis')
                threshold_steps = np.unique(ref)
            elif param.seg_type == 'prob':
                print('Probabilistic analysis')
                threshold_steps = np.arange(0, 1, param.step)

            for j in threshold_steps:
                if j == 0: continue
                if j >= 1:  # discrete eval
                    seg_binary = np.asarray(seg == j, dtype=np.float32)
                    ref_binary = np.asarray(ref == j, dtype=np.float32)
                elif j < 1:  # prob eval
                    seg_binary = np.asarray(seg >= j, dtype=np.float32)
                    ref_binary = np.asarray(ref >= 0.5, dtype=np.float32)
                if np.all(seg_binary == 0):
                    print("Empty foreground in thresholded binary image.")
                    continue
                PE = PairwiseMeasures(seg_binary, ref_binary,
                                      measures=MEASURES, num_neighbors=6,
                                      pixdim=voxel_sizes)
                fixed_fields = "{}, {}, {},".format(ref_name, seg_name, j)
                out_stream.write(fixed_fields + PE.to_string(
                    OUTPUT_FORMAT) + '\n')
