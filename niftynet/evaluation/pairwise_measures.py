# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
from scipy import ndimage

from niftynet.utilities.util_common import MorphologyOps, CacheFunctionOutput


class PairwiseMeasures(object):
    def __init__(self, seg_img, ref_img,
                 measures=None, num_neighbors=8, pixdim=(1, 1, 1),
                 empty=False, list_labels=None):

        self.m_dict = {
            'ref volume': (self.n_pos_ref, 'Volume (Ref)'),
            'seg volume': (self.n_pos_seg, 'Volume (Seg)'),
            'ref bg volume': (self.n_neg_ref, 'Volume (Ref bg)'),
            'seg bg volume': (self.n_neg_seg, 'Volume (Seg bg)'),
            'list_labels': (self.list_labels, 'List Labels Seg'),
            'fp': (self.fp, 'FP'),
            'fn': (self.fn, 'FN'),
            'tp': (self.tp, 'TP'),
            'tn': (self.tn, 'TN'),
            'n_intersection': (self.n_intersection, 'Intersection'),
            'n_union': (self.n_union, 'Union'),
            'sensitivity': (self.sensitivity, 'Sens'),
            'specificity': (self.specificity, 'Spec'),
            'accuracy': (self.accuracy, 'Acc'),
            'fpr': (self.false_positive_rate, 'FPR'),
            'ppv': (self.positive_predictive_values, 'PPV'),
            'npv': (self.negative_predictive_values, 'NPV'),
            'dice': (self.dice_score, 'Dice'),
            'IoU': (self.intersection_over_union, 'IoU'),
            'jaccard': (self.jaccard, 'Jaccard'),
            'informedness': (self.informedness, 'Informedness'),
            'markedness': (self.markedness, 'Markedness'),
            'vol_diff': (self.vol_diff, 'VolDiff'),
            'ave_dist': (self.measured_average_distance, 'AveDist'),
            'haus_dist': (self.measured_hausdorff_distance, 'HausDist'),
            'connected_elements': (self.connected_elements, 'TPc,FPc,FNc'),
            'outline_error': (self.outline_error, 'OER,OEFP,OEFN'),
            'detection_error': (self.detection_error, 'DE,DEFP,DEFN'),
            'com_dist': (self.com_dist, 'COM distance'),
            'com_ref': (self.com_ref, 'COM reference'),
            'com_seg': (self.com_seg, 'COM segmentation')
        }
        self.seg = seg_img
        self.ref = ref_img
        self.list_labels = list_labels
        self.flag_empty = empty
        self.measures = measures if measures is not None else self.m_dict
        self.neigh = num_neighbors
        self.pixdim = pixdim

    def check_binary(self):
        """
        Checks whether self.seg and self.ref are binary. This is to enable
        measurements such as 'false positives', which only have meaning in 
        the binary case (what is positive/negative for multiple class?)
        """

        is_seg_binary, is_ref_binary = [((x > 0.5) == x).all()
                                        for x in [self.seg, self.ref]]
        if (not is_ref_binary) or (not is_seg_binary):
            raise ValueError("The input segmentation/reference images"
                             " must be binary for this function.")

    def __FPmap(self):
        """
        This function calculates the false positive map from binary 
        segmentation and reference maps

        :return: FP map
        """
        self.check_binary()
        return np.asarray((self.seg - self.ref) > 0.0, dtype=np.float32)

    def __FNmap(self):
        """
        This function calculates the false negative map

        :return: FN map
        """
        self.check_binary()
        return np.asarray((self.ref - self.seg) > 0.0, dtype=np.float32)

    def __TPmap(self):
        """
        This function calculates the true positive map (i.e. how many 
        reference voxels are positive)

        :return: TP map
        """
        self.check_binary()
        return np.logical_and(self.ref > 0.5, self.seg > 0.5).astype(float)

    def __TNmap(self):
        """
        This function calculates the true negative map

        :return: TN map
        """
        self.check_binary()
        return np.logical_and(self.ref < 0.5, self.seg < 0.5).astype(float)

    def __union_map(self):
        """
        This function calculates the union map between segmentation and
        reference image

        :return: union map
        """
        self.check_binary()
        return np.logical_or(self.ref, self.seg).astype(float)

    def __intersection_map(self):
        """
        This function calculates the intersection between segmentation and
        reference image

        :return: intersection map
        """
        self.check_binary()
        return np.multiply(self.ref, self.seg)

    @CacheFunctionOutput
    def n_pos_ref(self):
        return np.sum(self.ref)

    @CacheFunctionOutput
    def n_neg_ref(self):
        self.check_binary()
        return np.sum(self.ref == 0)

    @CacheFunctionOutput
    def n_pos_seg(self):
        return np.sum(self.seg)

    @CacheFunctionOutput
    def n_neg_seg(self):
        return np.sum(1 - self.seg)

    @CacheFunctionOutput
    def fp(self):
        return np.sum(self.__FPmap())

    @CacheFunctionOutput
    def fn(self):
        return np.sum(self.__FNmap())

    @CacheFunctionOutput
    def tp(self):
        return np.sum(self.__TPmap())

    @CacheFunctionOutput
    def tn(self):
        return np.sum(self.__TNmap())

    @CacheFunctionOutput
    def n_intersection(self):
        return np.sum(self.__intersection_map())

    @CacheFunctionOutput
    def n_union(self):
        return np.sum(self.__union_map())

    def sensitivity(self):
        return self.tp() / self.n_pos_ref()

    def specificity(self):
        return self.tn() / self.n_neg_ref()

    def accuracy(self):
        return (self.tn() + self.tp()) / \
               (self.tn() + self.tp() + self.fn() + self.fp())

    def false_positive_rate(self):
        return self.fp() / self.n_neg_ref()

    def positive_predictive_values(self):
        if self.flag_empty:
            return -1
        return self.tp() / (self.tp() + self.fp())

    def negative_predictive_values(self):
        """
        This function calculates the negative predictive value ratio between
        the number of true negatives and the total number of negative elements

        :return:
        """
        return self.tn() / (self.fn() + self.tn())

    def dice_score(self):
        """
        This function returns the dice score coefficient between a reference
        and segmentation images

        :return: dice score
        """
        return 2 * self.tp() / np.sum(self.ref + self.seg)

    def intersection_over_union(self):
        """
        This function the intersection over union ratio - Definition of
        jaccard coefficient

        :return:
        """
        return self.n_intersection() / self.n_union()

    def jaccard(self):
        """
        This function returns the jaccard coefficient (defined as
        intersection over union)

        :return: jaccard coefficient
        """
        return self.intersection_over_union()

    def informedness(self):
        """
        This function calculates the informedness between the segmentation
        and the reference

        :return: informedness
        """
        return self.sensitivity() + self.specificity() - 1

    def markedness(self):
        """
        This functions calculates the markedness
        :return:
        """
        return self.positive_predictive_values() + \
               self.negative_predictive_values() - 1

    def com_dist(self):
        """
        This function calculates the euclidean distance between the centres
        of mass of the reference and segmentation.

        :return:
        """
        if self.flag_empty:
            return -1
        com_ref = ndimage.center_of_mass(self.ref)
        com_seg = ndimage.center_of_mass(self.seg)
        com_dist = np.sqrt(np.dot(np.square(np.asarray(com_ref) -
                                            np.asarray(com_seg)), np.square(
            self.pixdim)))
        return com_dist

    def com_ref(self):
        """
        This function calculates the centre of mass of the reference
        segmentation

        :return:
        """
        return ndimage.center_of_mass(self.ref) * np.array(self.pixdim)

    def com_seg(self):
        """
        This functions provides the centre of mass of the segmented element

        :return:
        """
        if self.flag_empty:
            return -1
        return ndimage.center_of_mass(self.seg)

    def list_labels(self):
        if self.list_labels is None:
            return ()
        return tuple(np.unique(self.list_labels))

    def vol_diff(self):
        """
        This function calculates the ratio of difference in volume between
        the reference and segmentation images.

        :return: vol_diff
        """
        return np.abs(self.n_pos_ref() - self.n_pos_seg()) / self.n_pos_ref()

    # @CacheFunctionOutput
    # def _boundaries_dist_mat(self):
    #     dist = DistanceMetric.get_metric('euclidean')
    #     border_ref = MorphologyOps(self.ref, self.neigh).border_map()
    #     border_seg = MorphologyOps(self.seg, self.neigh).border_map()
    #     coord_ref = np.multiply(np.argwhere(border_ref > 0), self.pixdim)
    #     coord_seg = np.multiply(np.argwhere(border_seg > 0), self.pixdim)
    #     pairwise_dist = dist.pairwise(coord_ref, coord_seg)
    #     return pairwise_dist

    @CacheFunctionOutput
    def border_distance(self):
        """
        This functions determines the map of distance from the borders of the
        segmentation and the reference and the border maps themselves

        :return: distance_border_ref, distance_border_seg, border_ref,
            border_seg
        """
        border_ref = MorphologyOps(self.ref, self.neigh).border_map()
        border_seg = MorphologyOps(self.seg, self.neigh).border_map()
        oppose_ref = 1 - self.ref
        oppose_seg = 1 - self.seg
        # euclidean distance transform
        distance_ref = ndimage.distance_transform_edt(oppose_ref)
        distance_seg = ndimage.distance_transform_edt(oppose_seg)
        distance_border_seg = border_ref * distance_seg
        distance_border_ref = border_seg * distance_ref
        return distance_border_ref, distance_border_seg, border_ref, border_seg

    def measured_distance(self):
        """
        This functions calculates the average symmetric distance and the
        hausdorff distance between a segmentation and a reference image

        :return: hausdorff distance and average symmetric distance
        """
        ref_border_dist, seg_border_dist, ref_border, \
            seg_border = self.border_distance()
        average_distance = (np.sum(ref_border_dist) + np.sum(
            seg_border_dist)) / (np.sum(self.ref + self.seg))
        hausdorff_distance = np.max(
            [np.max(ref_border_dist), np.max(seg_border_dist)])
        return hausdorff_distance, average_distance

    def measured_average_distance(self):
        """
        This function returns only the average distance when calculating the
        distances between segmentation and reference

        :return:
        """
        return self.measured_distance()[1]

    def measured_hausdorff_distance(self):
        """
        This function returns only the hausdorff distance when calculated the
        distances between segmentation and reference

        :return:
        """
        return self.measured_distance()[0]

    # def average_distance(self):
    #     pairwise_dist = self._boundaries_dist_mat()
    #     return (np.sum(np.min(pairwise_dist, 0)) + \
    #             np.sum(np.min(pairwise_dist, 1))) / \
    #            (np.sum(self.ref + self.seg))
    #
    # def hausdorff_distance(self):
    #     pairwise_dist = self._boundaries_dist_mat()
    #     return np.max((np.max(np.min(pairwise_dist, 0)),
    #                    np.max(np.min(pairwise_dist, 1))))

    @CacheFunctionOutput
    def _connected_components(self):
        """
        This function creates the maps of connected component for the
        reference and the segmentation image using the neighborhood defined
        in self.neigh

        :return:
            blobs_ref: connected labeling for the reference image,
            blobs_seg: connected labeling for the segmentation image,
            init: intersection between segmentation and reference
        """
        init = np.multiply(self.seg, self.ref)
        blobs_ref = MorphologyOps(self.ref, self.neigh).foreground_component()
        blobs_seg = MorphologyOps(self.seg, self.neigh).foreground_component()
        return blobs_ref, blobs_seg, init

    def connected_elements(self):
        """
        This function returns the number of FP FN and TP in terms of
        connected components.

        :return: Number of true positive connected components, Number of
            false positives connected components, Number of false negatives
            connected components
        """
        blobs_ref, blobs_seg, init = self._connected_components()
        list_blobs_ref = range(1, blobs_ref[1])
        list_blobs_seg = range(1, blobs_seg[1])
        mul_blobs_ref = np.multiply(blobs_ref[0], init)
        mul_blobs_seg = np.multiply(blobs_seg[0], init)
        list_TP_ref = np.unique(mul_blobs_ref[mul_blobs_ref > 0])
        list_TP_seg = np.unique(mul_blobs_seg[mul_blobs_seg > 0])

        list_FN = [x for x in list_blobs_ref if x not in list_TP_ref]
        list_FP = [x for x in list_blobs_seg if x not in list_TP_seg]
        return len(list_TP_ref), len(list_FP), len(list_FN)

    @CacheFunctionOutput
    def connected_errormaps(self):
        """
        This functions calculates the error maps from the connected components

        :return:
        """
        blobs_ref, blobs_seg, init = self._connected_components()
        list_blobs_ref = range(1, blobs_ref[1])
        list_blobs_seg = range(1, blobs_seg[1])
        mul_blobs_ref = np.multiply(blobs_ref[0], init)
        mul_blobs_seg = np.multiply(blobs_seg[0], init)
        list_TP_ref = np.unique(mul_blobs_ref[mul_blobs_ref > 0])
        list_TP_seg = np.unique(mul_blobs_seg[mul_blobs_seg > 0])

        list_FN = [x for x in list_blobs_ref if x not in list_TP_ref]
        list_FP = [x for x in list_blobs_seg if x not in list_TP_seg]
        # print(np.max(blobs_ref),np.max(blobs_seg))
        tpc_map = np.zeros_like(blobs_ref[0])
        fpc_map = np.zeros_like(blobs_ref[0])
        fnc_map = np.zeros_like(blobs_ref[0])
        for i in list_TP_ref:
            tpc_map[blobs_ref[0] == i] = 1
        for i in list_TP_seg:
            tpc_map[blobs_seg[0] == i] = 1
        for i in list_FN:
            fnc_map[blobs_ref[0] == i] = 1
        for i in list_FP:
            fpc_map[blobs_seg[0] == i] = 1
        return tpc_map, fnc_map, fpc_map

    def outline_error(self):
        """
        This function calculates the outline error as defined in Wack et al.

        :return: OER: Outline error ratio, OEFP: number of false positive
            outlier error voxels, OEFN: number of false negative outline error
            elements
        """
        TPcMap, _, _ = self.connected_errormaps()
        OEFMap = self.ref - np.multiply(TPcMap, self.seg)
        unique, counts = np.unique(OEFMap, return_counts=True)
        # print(counts)
        OEFN = counts[unique == 1]
        OEFP = counts[unique == -1]
        OEFN = 0 if len(OEFN) == 0 else OEFN[0]
        OEFP = 0 if len(OEFP) == 0 else OEFP[0]
        OER = 2 * (OEFN + OEFP) / (self.n_pos_seg() + self.n_pos_ref())
        return OER, OEFP, OEFN

    def detection_error(self):
        """
        This function calculates the volume of detection error as defined in
        Wack et al.

        :return: DE: Total volume of detection error, DEFP: Detection error
            false positives, DEFN: Detection error false negatives
        """
        TPcMap, FNcMap, FPcMap = self.connected_errormaps()
        DEFN = np.sum(FNcMap)
        DEFP = np.sum(FPcMap)
        return DEFN + DEFP, DEFP, DEFN

    def header_str(self):
        result_str = [self.m_dict[key][1] for key in self.measures]
        result_str = ',' + ','.join(result_str)
        return result_str

    def to_string(self, fmt='{:.4f}'):
        result_str = ""
        list_space = ['com_ref', 'com_seg', 'list_labels']
        for key in self.measures:
            result = self.m_dict[key][0]()
            if key in list_space:
                result_str += ' '.join(fmt.format(x) for x in result) \
                    if isinstance(result, tuple) else fmt.format(result)
            else:
                result_str += ','.join(fmt.format(x) for x in result) \
                    if isinstance(result, tuple) else fmt.format(result)
            result_str += ','
        return result_str[:-1]  # trim the last comma


class PairwiseMeasuresRegression(object):
    def __init__(self, reg_img, ref_img, measures=None):
        self.reg = reg_img
        self.ref = ref_img
        self.measures = measures

        self.m_dict = {
            'mse': (self.mse, 'MSE'),
            'rmse': (self.rmse, 'RMSE'),
            'mae': (self.mae, 'MAE'),
            'r2': (self.r2, 'R2')
        }

    def mse(self):
        return np.mean(np.square(self.reg - self.ref))

    def rmse(self):
        return np.sqrt(self.mse())

    def mae(self):
        return np.mean(np.abs(self.ref - self.reg))

    def r2(self):
        ref_var = np.sum(np.square(self.ref - np.mean(self.ref)))
        reg_var = np.sum(np.square(self.reg - np.mean(self.reg)))
        cov_refreg = np.sum(
            (self.reg - np.mean(self.reg)) * (self.ref - np.mean(
                self.ref)))
        return np.square(cov_refreg / np.sqrt(ref_var * reg_var + 0.00001))

    def header_str(self):
        result_str = [self.m_dict[key][1] for key in self.measures]
        result_str = ',' + ','.join(result_str)
        return result_str

    def to_string(self, fmt='{:.4f}'):
        result_str = ""
        for key in self.measures:
            result = self.m_dict[key][0]()
            result_str += ','.join(fmt.format(x) for x in result) \
                if isinstance(result, tuple) else fmt.format(result)
            result_str += ','
        return result_str[:-1]  # trim the last comma
