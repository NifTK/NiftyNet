from __future__ import absolute_import, print_function

import os.path

import nibabel as nib
import numpy as np

import niftynet.utilities.csv_table as csv_table
from niftynet.evaluation.pairwise_measures import PairwiseMeasures
from niftynet.utilities.misc_common import MorphologyOps
MEASURES = (
    'ref volume', 'seg volume',
    'tp', 'fp', 'fn',
    'connected_elements', 'vol_diff',
    'outline_error', 'detection_error',
    'fpr', 'ppv', 'npv', 'sensitivity', 'specificity',
    'accuracy', 'jaccard', 'dice', 'ave_dist', 'haus_dist'
)
MEASURES_LABELS = ('ref volume','seg volume','list_labels','tp','fp','fn',
                 'vol_diff','fpr','ppv','sensitivity','specificity',
                 'accuracy','jaccard','dice','ave_dist','haus_dist',
                   'com_dist','com_ref','com_seg')

# MEASURES_NEW = ('ref volume', 'seg volume', 'tp', 'fp', 'fn', 'outline_error',
#             'detection_error', 'dice')
OUTPUT_FORMAT = '{:4f}'
OUTPUT_FILE_PREFIX = 'PairwiseMeasure'


def run(param, csv_dict):
    # output
    out_name = '{}_{}_{}_{}.csv'.format(
        OUTPUT_FILE_PREFIX,
        os.path.split(param.ref_dir)[1],
        os.path.split(param.seg_dir)[1],
        param.save_name)
    iteration = 0
    while os.path.exists(os.path.join(param.save_csv_dir, out_name)):
        iteration += 1
        out_name = '{}_{}_{}_{}_{}.csv'.format(
        OUTPUT_FILE_PREFIX,
        os.path.split(param.ref_dir)[1],
        os.path.split(param.seg_dir)[1],
        param.save_name, str(iteration))

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
    with open(os.path.join(param.save_csv_dir, out_name), 'w+',
              0) as out_stream:
        # a trivial PairwiseMeasures obj to produce header_str
        if param.seg_type == 'discrete':
            m_headers = PairwiseMeasures(0, 0,
                                         measures=MEASURES_LABELS).header_str()
            out_stream.write("Name (ref), Name (seg), Label" + m_headers + '\n')
            measures_fin = MEASURES_LABELS
        else:
            m_headers = PairwiseMeasures(0, 0,
                                         measures=MEASURES).header_str()
            out_stream.write("Name (ref), Name (seg), Label" + m_headers + '\n')
            measures_fin = MEASURES

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
            # Create and save nii files of map of differences (FP FN TP OEMap
            #  DE if flag_save_map is on and binary segmentation


            flag_createlab = False
            if (param.seg_type == 'discrete') and (np.max(seg) <= 1):
                flag_createlab = True
                seg = np.asarray(seg >= param.threshold)
                ref = np.asarray(ref >= param.threshold)
                blob_ref = MorphologyOps(ref, 6).foreground_component()
                ref = blob_ref[0]
                blob_seg = MorphologyOps(seg, 6).foreground_component()
                seg = blob_seg[0]
                if param.save_discrete:
                    label_ref_nii = nib.Nifti1Image(ref, ref_nii.affine)
                    label_seg_nii = nib.Nifti1Image(seg, seg_nii.affine)
                    name_ref_label = os.path.join(param.save_csv_dir,
                                                  'LabelsRef_'+os.path.split(
                                                      ref_name)[1])
                    name_seg_label = os.path.join(param.save_csv_dir,
                                                  'LabelsSeg_'+os.path.split(
                                                      seg_name)[1])
                    nib.save(label_ref_nii, name_ref_label)
                    nib.save(label_seg_nii, name_seg_label)

                #     and (len(np.unique(seg)) > 2):
                # print('Non-integer class labels for discrete analysis')
                # print('Thresholding to binary map with threshold: {}'.format(
                #     param.threshold))
                # seg = np.asarray(seg >= param.threshold, dtype=np.int8)


            ## TODO: user specifies how to convert seg -> seg_binary
            if param.seg_type == 'discrete':
                print('Discrete analysis')
                threshold_steps = np.unique(ref)
            elif param.seg_type == 'prob':
                print('Probabilistic analysis')
                threshold_steps = np.arange(0, 1, param.step)
            else:
                print('Binary analysis')
                threshold_steps = [param.threshold]

            for j in threshold_steps:
                if j == 0: continue
                list_labels_seg = []
                if j >= 1:
                    if not flag_createlab:  # discrete eval with same
                    # labels
                        seg_binary = np.asarray(seg == j, dtype=np.float32)
                        ref_binary = np.asarray(ref == j, dtype=np.float32)


                    else: # different segmentations with connected components
                        #  (for instance lesion segmentation)
                        ref_binary = np.asarray(ref == j, dtype=np.float32)
                        seg_matched = np.multiply(ref_binary, seg)
                        list_labels_seg = np.unique(seg_matched)
                        seg_binary = np.zeros_like(ref_binary)
                        for l in list_labels_seg:
                            if l > 0:
                                seg_temp = np.asarray(seg == l)
                                seg_binary = seg_binary + seg_temp
                        print(np.sum(seg_binary))

                elif j < 1:  # prob or binary eval
                    seg_binary = np.asarray(seg >= j, dtype=np.float32)
                    ref_binary = np.asarray(ref >= 0.5, dtype=np.float32)
                    if param.save_maps and param.seg_type == 'binary':
                        #Creation of the error maps per type and saving
                        temp_pe = PairwiseMeasures(seg_binary, ref_binary,
                                                   measures=(
                                                       'outline_error'),
                                                   num_neighbors=6,
                                                   pixdim=voxel_sizes)
                        tp_map, fp_map, fn_map = \
                            temp_pe.connected_errormaps()
                        intersection = np.multiply(seg_binary, ref_binary)
                        oefp_map = np.multiply(tp_map, seg_binary) - \
                                   intersection
                        oefn_map = np.multiply(tp_map, ref_binary) - \
                                   intersection
                        oefp_nii = nib.Nifti1Image(oefp_map, ref_nii.affine)
                        oefn_nii = nib.Nifti1Image(oefn_map, ref_nii.affine)
                        tp_nii = nib.Nifti1Image(intersection,
                                                 ref_nii.affine)
                        defp_nii = nib.Nifti1Image(fp_map, ref_nii.affine)
                        defn_nii = nib.Nifti1Image(fn_map, ref_nii.affine)
                        defn_name = os.path.join(param.save_csv_dir,
                                                 param.save_name +
                                                 '_DEFN_' + os.path.split(
                                                     seg_name)[1])
                        defp_name = os.path.join(param.save_csv_dir,
                                                 param.save_name +
                                                 '_DEFP_' + os.path.split(
                                                     seg_name)[1])
                        oefn_name = os.path.join(param.save_csv_dir,
                                                 param.save_name +
                                                 '_OEFN_' + os.path.split(
                                                     seg_name)[1])
                        oefp_name = os.path.join(param.save_csv_dir,
                                                 param.save_name +
                                                 '_OEFP_' + os.path.split(
                                                     seg_name)[1])
                        tp_name = os.path.join(param.save_csv_dir,
                                               param.save_name +
                                               '_TP_' + os.path.split(
                                                   seg_name)[1])

                        nib.save(oefn_nii, oefn_name)
                        nib.save(oefp_nii, oefp_name)
                        nib.save(tp_nii, tp_name)
                        nib.save(defp_nii, defp_name)
                        nib.save(defn_nii, defn_name)
                if np.all(seg_binary == 0): # Have to put default results.
                    print("Empty foreground in thresholded binary image.")
                    PE = PairwiseMeasures(seg_binary, ref_binary,
                                          measures=measures_fin,
                                          num_neighbors=6,
                                          pixdim=voxel_sizes, empty=True)
                else:
                    PE = PairwiseMeasures(seg_binary, ref_binary,
                                      measures=measures_fin, num_neighbors=6,
                                      pixdim=voxel_sizes,
                                          list_labels=list_labels_seg)
                if len(list_labels_seg) > 0 and 'list_labels' in measures_fin:
                    PE.list_labels = list_labels_seg
                fixed_fields = "{}, {}, {},".format(ref_name, seg_name, j)
                out_stream.write(fixed_fields + PE.to_string(
                    OUTPUT_FORMAT) + '\n')
                out_stream.flush()
                os.fsync(out_stream.fileno())
