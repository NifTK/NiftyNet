import numpy as np
from sklearn.neighbors import DistanceMetric
from util import MorphologyOps
from util import SimpleCache


class PairwiseMeasures(object):
    def __init__(self, seg_img, ref_img,
            measures=None, num_neighbors=8, pixdim=[1, 1, 1]):

        self.m_dict = {
            'ref volume': (self.n_pos_ref, 'Volume (Ref)'),
            'seg volume': (self.n_pos_seg, 'Volume (Seg)'),
            'ref bg volume': (self.n_neg_ref, 'Volume (Ref bg)'),
            'seg bg volume': (self.n_neg_seg, 'Volume (Seg bg)'),
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
            'ave_dist': (self.average_distance, 'AveDist'),
            'haus_dist': (self.hausdorff_distance, 'HausDist'),
            'connected_elements': (self.connected_elements, 'TPc,FPc,FNc'),
            'outline_error': (self.outline_error, 'OER,OEFP,OEFN'),
            'detection_error': (self.detection_error, 'DE,DEFP,DEFN'),
        }
        self.seg = seg_img
        self.ref = ref_img
        self.measures = measures if measures is not None else self.m_dict
        self.neigh = num_neighbors
        self.pixdim = pixdim

    def __FPmap(self):
        return np.asarray((self.seg - self.ref) > 0.0, dtype=np.float32)

    def __FNmap(self):
        return np.asarray((self.ref - self.seg) > 0.0, dtype=np.float32)

    def __TPmap(self):
        return np.asarray((self.ref + self.seg) > 1.0, dtype=np.float32)

    def __TNmap(self):
        return np.asarray((self.ref + self.seg) < 0.5, dtype=np.float32)

    def __union_map(self):
        return np.asarray((self.ref + self.seg) > 0.5, dtype=np.float32)

    def __intersection_map(self):
        return np.multiply(self.ref, self.seg)

    @SimpleCache
    def n_pos_ref(self):
        return np.sum(self.ref)

    @SimpleCache
    def n_neg_ref(self):
        return np.sum(1 - self.ref)

    @SimpleCache
    def n_pos_seg(self):
        return np.sum(self.seg)

    @SimpleCache
    def n_neg_seg(self):
        return np.sum(1 - self.seg)

    @SimpleCache
    def fp(self):
        return np.sum(self.__FPmap())

    @SimpleCache
    def fn(self):
        return np.sum(self.__FNmap())

    @SimpleCache
    def tp(self):
        return np.sum(self.__TPmap())

    @SimpleCache
    def tn(self):
        return np.sum(self.__TNmap())

    @SimpleCache
    def n_intersection(self):
        return np.sum(self.__intersection_map())

    @SimpleCache
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
        return self.tp() / (self.tp() + self.fp())

    def negative_predictive_values(self):
        return self.tn() / (self.fn() + self.tn())

    def dice_score(self):
        return 2 * self.tp() / np.sum(self.ref + self.seg)

    def intersection_over_union(self):
        return self.n_intersection() / self.n_union()

    def jaccard(self):
        return self.n_intersection() / self.n_union()

    def informedness(self):
        return self.sensitivity() + self.specificity() - 1

    def markedness(self):
        return self.positive_predictive_values() + \
                self.negative_predictive_values() - 1

    def vol_diff(self):
        return np.abs(self.n_pos_ref() - self.n_pos_seg()) / self.n_pos_ref()

    @SimpleCache
    def _boundaries_dist_mat(self):
        dist = DistanceMetric.get_metric('euclidean')
        border_ref = MorphologyOps(self.ref, self.neigh).border_map()
        border_seg = MorphologyOps(self.seg, self.neigh).border_map()
        coord_ref = np.multiply(np.argwhere(border_ref > 0), self.pixdim)
        coord_seg = np.multiply(np.argwhere(border_seg > 0), self.pixdim)
        pairwise_dist = dist.pairwise(coord_ref, coord_seg)
        return pairwise_dist

    def average_distance(self):
        pairwise_dist = self._boundaries_dist_mat()
        return (np.sum(np.min(pairwise_dist, 0)) + \
                np.sum(np.min(pairwise_dist, 1))) / \
                (np.sum(self.ref + self.seg))

    def hausdorff_distance(self):
        pairwise_dist = self._boundaries_dist_mat()
        return np.max((np.max(np.min(pairwise_dist, 0)),
            np.max(np.min(pairwise_dist,1))))

    @SimpleCache
    def _connected_components(self):
        init = np.multiply(self.seg, self.ref)
        blobs_ref = MorphologyOps(self.ref, self.neigh).foreground_component()
        blobs_seg = MorphologyOps(self.seg, self.neigh).foreground_component()
        return blobs_ref, blobs_seg, init

    def connected_elements(self):
        blobs_ref, blobs_seg, init = self._connected_components()
        list_blobs_ref = np.unique(blobs_ref[blobs_ref > 0])
        list_blobs_seg = np.unique(blobs_seg[blobs_seg > 0])
        mul_blobs_ref = np.multiply(blobs_ref, init)
        mul_blobs_seg = np.multiply(blobs_seg, init)
        list_TP_ref = np.unique(mul_blobs_ref[mul_blobs_ref > 0])
        list_TP_seg = np.unique(mul_blobs_seg[mul_blobs_seg > 0])

        list_FN = [x for x in list_blobs_ref if x not in list_TP_ref]
        list_FP = [x for x in list_blobs_seg if x not in list_TP_seg]
        return len(list_TP_ref), len(list_FP), len(list_FN)

    @SimpleCache
    def connected_errormaps(self):
        blobs_ref, blobs_seg, init = self._connected_components()
        list_blobs_ref = np.unique(blobs_ref[blobs_ref > 0])
        list_blobs_seg = np.unique(blobs_seg[blobs_seg > 0])
        mul_blobs_ref = np.multiply(blobs_ref, init)
        mul_blobs_seg = np.multiply(blobs_seg, init)
        list_TP_ref = np.unique(mul_blobs_ref[mul_blobs_ref>0])
        list_TP_seg = np.unique(mul_blobs_seg[mul_blobs_seg>0])

        list_FN = [x for x in list_blobs_ref if x not in list_TP_ref]
        list_FP = [x for x in list_blobs_seg if x not in list_TP_seg]
        #print(np.max(blobs_ref),np.max(blobs_seg))
        tpc_map = np.zeros_like(blobs_ref)
        fpc_map = np.zeros_like(blobs_ref)
        fnc_map = np.zeros_like(blobs_ref)
        for i in list_TP_ref:
            tpc_map[blobs_ref == i] = 1
        for i in list_TP_seg:
            tpc_map[blobs_seg == i] = 1
        for i in list_FN:
            fnc_map[blobs_ref == i] = 1
        for i in list_FP:
            fpc_map[blobs_seg == i] = 1
        return tpc_map, fnc_map, fpc_map

    def outline_error(self):
        TPcMap, _, _ = self.connected_errormaps()
        OEFMap = self.ref - np.multiply(TPcMap, self.seg)
        unique, counts = np.unique(OEFMap, return_counts=True)
        #print(counts)
        OEFN = counts[unique == 1]
        OEFP = counts[unique == -1]
        OEFN = 0 if len(OEFN) == 0 else OEFN[0]
        OEFP = 0 if len(OEFP) == 0 else OEFP[0]
        OER = 2*(OEFN + OEFP) / (self.n_pos_seg() + self.n_pos_ref())
        return OER, OEFP, OEFN

    def detection_error(self):
        TPcMap, FNcMap, FPcMap = self.connected_errormaps()
        DEFN = np.sum(FNcMap)
        DEFP = np.sum(FPcMap)
        return DEFN+DEFP, DEFP, DEFN

    def header_str(self):
        result_str = ""
        for key in self.measures:
            header = self.m_dict[key][1]
            result_str += ','.join(x for x in header) if \
                    isinstance(header, tuple) else header
            result_str += ','
        return result_str[:-1]

    def to_string(self, fmt='{:.4f}'):
        result_str = ""
        for key in self.measures:
            result = self.m_dict[key][0]()
            result_str += ','.join(fmt.format(x) for x in result) \
                    if isinstance(result, tuple) else fmt.format(result)
            result_str += ','
        return result_str[:-1]  # trim the last comma
