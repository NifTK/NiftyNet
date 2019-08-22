# -*- coding: utf-8 -*-
"""
This module defines built-in evaluation functions for segmentation applications

Segmentations can be evaluated at several scales:
'foreground' refering to metrics computed once for a foreground label
'label' refering to metrics computed once for each label (including background)
'cc' referring to metrics computed once for each connected component set
    Connected components are defined by one-or-more connected
    components on the reference segmentation and one-or-more connected
    components on the infered segmentation.
    These sets are defined by a cc_func. Currently
    this is hard coded to be union_of_seg_for_each_ref_cc which takes each
    connected component on the reference segmentation and all connected
    components on the infered segmentation with any overlap. This will
    eventually be made into a factory option for different cc set definitions

Overlap and distance measures can be computed at each of these levels by
deriving from PerComponentEvaluation, which handles the logic of identifying
which comparisons need to be done for each scale.

Overlap and distance measures are computed in two convenience functions
(compute_many_overlap_metrics and compute_many_distance_metrics) and wrapped
by Evaluation classes
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from scipy import ndimage

from niftynet.evaluation.base_evaluations import CachedSubanalysisEvaluation
from niftynet.utilities.util_common import MorphologyOps, \
    CachedFunction, CachedFunctionByID
from niftynet.evaluation.base_evaluator import ScalarAggregator


class PerComponentEvaluation(CachedSubanalysisEvaluation):
    """
    This class represents evaluations performed on binary segmentation
    components computed per label or per connected component. It encodes the
    generation of evaluation tasks. Derived classes should define the
    metric_name constant and the function metric_from_binarized()
    """

    def subanalyses(self, subject_id, data):
        analyses = self.app_param.evaluation_units.split(',')
        tasks = []
        for analysis in analyses:
            if analysis in ['foreground', 'label']:
                labels = list(range(self.app_param.num_classes))
                if analysis == 'foreground':
                    labels.remove(0)
                for label in labels:
                    tasks.append({'subject_id': subject_id, 'label': label})
            elif analysis in ['cc']:
                cc_seg, cc_ref = \
                    connected_components(data['inferred'], data['label'],
                                         self.app_param.output_prob)
                cc_func = union_of_seg_for_each_ref_cc  # TODO make into factory
                conncomps = cc_func(cc_seg, cc_ref)
                for conncomp in conncomps:
                    tasks.append({'subject_id': subject_id,
                                  'cc_labels': conncomps[conncomp]})
                    # TODO save an index image from blobs_ref[0]
        return tasks

    def layer_op(self, subject_id, data, task):
        # We use a cached binarizer function so that the binarized
        # segmentation have the same python id
        if 'label' in task:
            binarizer = cached_label_binarizer(task['label'],
                                               self.app_param.output_prob)
            seg, ref = binarizer(data)
            metric_dict = {'subject_id': subject_id, 'label': task['label']}
            metric_dict.update(self.metric_dict_from_binarized(seg, ref))
            pdf = pd.DataFrame.from_records([metric_dict], ('subject_id', 'label'))
            return [pdf]
        elif 'cc_labels' in task:
            binarizer = cached_cc_binarizer(task['cc_labels'],
                                            self.app_param.output_prob)
            seg, ref = binarizer(data)
            r_str = '&'.join([str(l) for l in task['cc_labels'][1]])
            s_str = '&'.join([str(l) for l in task['cc_labels'][0]])
            cc_id = 'r%s_s%s' % (r_str, s_str)
            metric_dict = {'subject_id': subject_id, 'cc_id': cc_id}
            metric_dict.update(self.metric_dict_from_binarized(seg, ref))
            pdf = pd.DataFrame.from_records([metric_dict], ('subject_id', 'cc_id'))
            return [pdf]
        return []


    def metric_dict_from_binarized(self, seg, ref):
        """
        Computes a metric from a binarized mask
        :param seg: numpy array with binary mask from inferred segmentation
        :param ref: numpy array with binary mask from reference segmentation
        :return: a dictionary of metric_name:metric_value
        """
        raise NotImplementedError('Not implemented in abstract base class')


class PerComponentScalarEvaluation(PerComponentEvaluation):
    """ This class simplifies the implementation when the metric just returns a
    single scalar with the same name as the class name"""
    def __init__(self, *args, **kwargs):
        super(PerComponentScalarEvaluation, self).__init__(*args,
                                                           **kwargs)
        self.metric_name = self.__class__.__name__

    def metric_dict_from_binarized(self, seg, ref):
        """ Wrap computed metric in dictionary for parent class """
        metric_value = self.metric_from_binarized(seg, ref)
        return {self.metric_name: metric_value}

    def metric_from_binarized(self, seg, ref):
        """
        Computer scalar metric value
        :param seg: numpy array with binary mask from inferred segmentation
        :param ref: numpy array with binary mask from reference segmentation
        :return: scalar metric value
        """

    def get_aggregations(self):
        aggregations = []
        analyses = self.app_param.evaluation_units.split(',')
        for analysis in analyses:
            if analysis in ['foreground', 'label']:
                mean_agg = ScalarAggregator(self.metric_name,
                                            ('subject_id', 'label'),
                                            ('label',), np.mean,
                                            'mean_' + self.metric_name)
                std_agg = ScalarAggregator(self.metric_name,
                                           ('subject_id', 'label'),
                                           ('label',), np.std,
                                           'stdev_' + self.metric_name)
                aggregations.extend([mean_agg, std_agg])
            elif analysis in ['cc']:
                pass
        return aggregations

class BuiltinOverlapEvaluation(PerComponentScalarEvaluation):
    """
    Wrapper class to encode many similar overlap metrics that can be computed
    from a confusion matrix
    Metrics computed in compute_many_overlap_metrics can be wrapped by
    overriding self.metric_name
    """
    def metric_from_binarized(self, seg, ref):
        """
        Computes a metric from a binarized mask by computing a confusion
        matrix and then delegating the metric computation
        :param seg: numpy array with binary mask from inferred segmentation
        :param ref: numpy array with binary mask from reference segmentation
        :return: scalar metric value
        """
        lnot = np.logical_not
        land = np.logical_and
        conf_mat = np.array([[np.sum(land(lnot(seg), lnot(ref))),
                              np.sum(land(lnot(seg), (ref)))],
                             [np.sum(land((seg), lnot(ref))),
                              np.sum(land((seg), (ref)))]])
        return self.metric_from_confusion_matrix(conf_mat)

    def metric_from_confusion_matrix(self, confusion_matrix):
        """
        Compute metrics from a 2x2 confusion matrix
        :param confusion_matrix: 2x2 numpy array
        :return: scalar metric value
        """


#pylint: disable=missing-docstring,invalid-name
class n_pos_ref(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[0, 1] + M[1, 1]


class n_neg_ref(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[0, 0] + M[1, 0]


class n_pos_seg(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[1, 0] + M[1, 1]


class n_neg_seg(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[0, 0] + M[0, 1]


class fp(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[1, 0]


class fn(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[0, 1]


class tp(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[1, 1]


class tn(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[0, 0]


class n_intersection(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[1, 1]


class n_union(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[0, 1] + M[1, 0] + M[1, 1]


class specificity(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[0, 0] / (M[0, 0] + M[1, 0])


class sensitivity(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[1, 1] / (M[0, 1] + M[1, 1])


class accuracy(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return (M[1, 1] + M[0, 0]) / sum(sum(M))


class false_positive_rate(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[1, 0] / (M[0, 0] + M[1, 0])


class positive_predictive_values(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[1, 1] / (M[1, 0] + M[1, 1])


class negative_predictive_values(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[0, 0] / (M[0, 0] + M[0, 1])


class dice(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return 2 * M[1, 1] / (M[1, 1] * 2 + M[1, 0] + M[0, 1])


Dice = dice


class jaccard(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[1, 1] / (M[0, 1] + M[1, 0] + M[1, 1])


intersection_over_union = jaccard
Jaccard = jaccard


class informedness(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[1, 1] / (M[0, 1] + M[1, 1]) + \
                             M[0, 0] / (M[0, 0] + M[1, 0]) - 1


class markedness(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return M[1, 1] / (M[1, 0] + M[1, 1]) + \
                           M[0, 0] / (M[0, 0] + M[0, 1]) - 1


class vol_diff(BuiltinOverlapEvaluation):
    def metric_from_confusion_matrix(self, M):
        return (M[1, 1] + M[1, 0]) / (M[0, 1] + M[1, 1])


# Distance metrics as e.g. in 10.3978/j.issn.2223-4292.2015.08.02
class average_distance(PerComponentScalarEvaluation):
    def metric_from_binarized(self, seg, ref):
        ref_border_dist, seg_border_dist = border_distance(seg, ref, 8)
        border_ref, border_seg = borders(seg, ref, 8)
        return (np.sum(ref_border_dist) + np.sum(
            seg_border_dist)) / (np.sum(border_ref + border_seg))


class hausdorff_distance(PerComponentScalarEvaluation):
    def metric_from_binarized(self, seg, ref):
        ref_border_dist, seg_border_dist = border_distance(seg, ref, 8)
        return np.max([np.max(ref_border_dist), np.max(seg_border_dist)])


class hausdorff95_distance(PerComponentScalarEvaluation):
    def metric_from_binarized(self, seg, ref):
        ref_border_dist, seg_border_dist = border_distance(seg, ref, 8)
        border_ref, border_seg = borders(seg, ref, 8)
        seg_values = ref_border_dist[border_seg > 0]
        ref_values = seg_border_dist[border_ref > 0]
        if seg_values.size == 0 or ref_values.size == 0:
            return np.nan
        return np.max([np.percentile(seg_values, 95),
                       np.percentile(ref_values, 95)])


#pylint: enable=missing-docstring,invalid-name
# Helper functions
@CachedFunction
def cached_label_binarizer(label, output_prob):
    """
    This class returns a function for binarizing an inferred segmentation
    for a specified label.
    This function is carefully designed to allow caching of unhashable numpy
    objects. Specifically, each call to cached_label_binarizer with the same
    (by-value) parameters will return the same (by python id) function
    object. This enables two calls to
    cached_label_binarizer(...)(numpy_object_1)
    with the same parameters from different metrics to use the cached result
    :param label:  Which label to make foreground in the binary mask
    :param output_prob: Is the segmentation probabilistic (if so,
    argmax is used first to compute a label map)
    :return: a function for computing a binary label map
    """
    @CachedFunctionByID
    def binarizer(data):
        """
        This function binarizes a segmentation based on a specified
        label (defined by outer function)
        :param data: a data dictionary as built by ImageReader
        :return: a numpy array representing a binary label map
        """
        if output_prob:
            out = np.argmax(data['inferred'], -1)
        else:
            out = data['inferred']
        return out == label, data['label'] == label

    return binarizer


@CachedFunction
def cached_cc_binarizer(cc_labels, output_prob):
    """
    This class returns a function for binarizing inferred and reference
    segmentations for a specified connected component set.
    This function is carefully designed to allow caching of unhashable numpy
    objects. Specifically, each call to cached_label_binarizer with the same
    (by-value) parameters will return the same (by python id) function
    object. This enables two calls to
    cached_cc_binarizer(...)(numpy_object_1)
    with the same parameters from different metrics to use the cached result
    :param cc_labels:  [seg_label_list, ref_label_list] where each is a
        list of values to be considered foreground for this cc set
    :param output_prob: Is the segmentation probabilistic (if so,
        argmax is used first to compute a label map)
    :return: a function for computing a binary label map pair

    """
    @CachedFunctionByID
    def binarizer(data):
        """
        This function binarizes a multi-object segmentation and reference
        into a specified connected component set (defined by outer function)
        :param data: a data dictionary as built by ImageReader
        :return: two numpy arrays representing binary masks (from
        inferred and reference segmentations) for a connected component set
        """
        cc_func = connected_components
        cc_seg, cc_ref = cc_func(data['inferred'], data['label'], output_prob)
        cc_seg_in = np.zeros_like(cc_seg[0])
        cc_ref_in = np.zeros_like(cc_ref[0])
        for i in cc_labels[0]:
            cc_seg_in[cc_seg[0] == i] = 1
        for i in cc_labels[1]:
            cc_ref_in[cc_ref[0] == i] = 1
        return cc_seg_in, cc_ref_in

    return binarizer

def union_of_seg_for_each_ref_cc(blobs_seg, blobs_ref):
    """
    Constructs connected component sets to compute metrics for. Each
    reference connected component is paired with the union of inferred
    segmentation connected components with any overlap
    :param blobs_seg: tuple (numbered_cc_array, number_of_ccs)
    :param blobs_ref: tuple (numbered_cc_array, number_of_ccs)
    :return: dictionary {label:(ref_label_list, seg_label_list)}
    """
    keys = {}
    for cc_id in range(1, blobs_ref[1] + 1):
        seg_idx = list(np.unique(blobs_seg[0][blobs_ref[0] == cc_id]))
        if 0 in seg_idx:
            seg_idx.remove(0)
        key = 'r' + str(cc_id) + '_c' + '_'.join([str(s) for s in seg_idx])
        keys[key] = ((cc_id,), tuple(seg_idx))
    return keys


@CachedFunctionByID
def borders(seg, ref, neigh=8):
    """
    This function determines the points that lie on the border of the
    inferred and reference segmentations
    :param seg: numpy array with binary mask from inferred segmentation
    :param ref: numpy array with binary mask from reference segmentation
    :param neigh: connectivity 4 or 8
    :return: numpy arrays of reference and inferred segmentation borders
    """
    border_ref = MorphologyOps(ref[:, :, :, 0, 0], neigh).border_map()
    border_seg = MorphologyOps(seg[:, :, :, 0, 0], neigh).border_map()
    return border_ref, border_seg


@CachedFunctionByID
def border_distance(seg, ref, neigh=8):
    """
    This functions determines the distance at each seg border point to the
    nearest ref border point and vice versa
    :param seg: numpy array with binary mask from inferred segmentation
    :param ref: numpy array with binary mask from reference segmentation
    :param neigh: connectivity 4 or 8
    :return: numpy arrays for distance_from_ref_border, distance_from
    seg_border
    """
    border_ref, border_seg = borders(seg, ref, neigh)
    distance_ref = ndimage.distance_transform_edt(1 - border_ref)
    distance_seg = ndimage.distance_transform_edt(1 - border_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg


@CachedFunctionByID
def connected_components(seg, ref, output_prob, neigh=8):
    """
    Numbers connected components in the reference and inferred segmentations
    :param seg: numpy array with binary mask from inferred segmentation
    :param ref: numpy array with binary mask from reference segmentation
    :param output_prob: Is the segmentation probabilistic (if so,
    argmax is used first to compute a label map)
    :param neigh: connectivity 4 or 8
    :return: (cc_map_ref, count) numbered connected components from reference
    :return: (cc_map_seg, count) numbered connected components from inferred
    """
    if output_prob:
        seg = np.argmax(seg, -1)
    blobs_ref = MorphologyOps(ref[:, :, :, 0, 0], neigh).foreground_component()
    blobs_seg = MorphologyOps(seg[:, :, :, 0, 0], neigh).foreground_component()

    return (blobs_ref[0][:, :, :, np.newaxis, np.newaxis], blobs_ref[1]), \
           (blobs_seg[0][:, :, :, np.newaxis, np.newaxis], blobs_seg[1]),


# TODO
# per subject connected component related metrics
# 'connected_elements': (self.connected_elements, 'TPc,FPc,FNc'),
# 'outline_error': (self.outline_error, 'OER,OEFP,OEFN'),
# 'detection_error': (self.detection_error, 'DE,DEFP,DEFN'),
# list_labels
# list connected components
# TODO image_map outputs
