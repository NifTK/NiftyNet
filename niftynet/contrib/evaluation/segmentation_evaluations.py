"""
This module holds built-in segmentation evaluations without tests
"""

import os

import numpy as np
import pandas as pd
from scipy import ndimage

from niftynet.evaluation.base_evaluations import BaseEvaluation
from niftynet.evaluation.segmentation_evaluations import \
    PerComponentEvaluation, connected_components, cached_label_binarizer, \
    union_of_seg_for_each_ref_cc
from niftynet.io.misc_io import save_data_array


class com_ref(PerComponentEvaluation):
    """
    Computes the centers of mass of each component in the reference standard
    """

    def metric_from_binarized(self, seg, ref):
        """
        :param seg: numpy array with binary mask from inferred segmentation
        :param ref: numpy array with binary mask from reference segmentation
        :return: dict of centers of mass in each axis
        """
        return {d: 'com_ref_' + x
                for d, x in zip('XYZ', ndimage.center_of_mass(ref))}


class ErrorMapsCC(BaseEvaluation):
    """
    Create 3 maps of connected component detection:
    tpc_map shows each detected ref cc (having at least on seg cc that
    overlaps) and the union of all overlapping seg ccs
    fnc_map shows all ref ccs that were not detected
    fpc_map shows all seg ccs that did not overlap any ref ccs
    Note we currently arbitrarily limit image generation to binary problems
    """

    def layer_op(self, subject_id, data):
        analyses = self.app_param.evaluation_units.split(',')
        if 'label' not in analyses and 'foreground' not in analyses:
            raise ValueError('ErrorMaps work with label or foreground '
                             'analyses only')
        if self.app_param.num_classes > 2:
            raise ValueError('ErrorMaps work with binary segmentations only')

        binarizer = cached_label_binarizer(1, self.app_param.output_prob)
        seg, ref = binarizer(data)
        cc_func = connected_components
        cc_seg, cc_ref = cc_func(seg, ref, self.app_param.output_prob)

        cc_aggregator = union_of_seg_for_each_ref_cc
        ccs = cc_aggregator(cc_ref, cc_seg)

        tp_seg_labels = set(s for seg_l, ref_l in ccs for s in seg_l)
        tp_ref_labels = set(r for seg_l, ref_l in ccs for r in ref_l if len(
            seg_l))
        fn_ref_labels = set(range(1, cc_ref[1])) - tp_ref_labels
        fp_seg_labels = set(range(1, cc_seg[1])) - tp_seg_labels
        maps = {}
        maps['tpc_map'] = np.logical_or(cc_seg in tp_seg_labels,
                                        cc_ref in tp_ref_labels)
        maps['fnc_map'] = cc_ref in fn_ref_labels
        maps['fpc_map'] = cc_seg in fp_seg_labels

        image_idx = self.reader.get_image_index(subject_id)
        file_path = os.path.join(self.eval_param.save_csv_dir, 'images')
        out = {'subject_id': subject_id}
        for key in maps:
            out[key] = os.path.join(file_path, subject_id + '_' + key + '.nii')
            save_data_array(file_path,
                            subject_id + '_' + key + '.nii',
                            maps[key],
                            self.reader.output_list[image_idx]['label'], 0)
        pdf = pd.DataFrame.from_records([out], ('subject_id',))
        return [pdf]
