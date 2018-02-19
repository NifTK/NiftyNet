# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from argparse import Namespace
import tensorflow as tf
import numpy as np

from niftynet.evaluation.pairwise_measures import PairwiseMeasures
from niftynet.utilities.util_common import MorphologyOps
from niftynet.evaluation.segmentation_evaluator import SegmentationEvaluator
import niftynet.evaluation.segmentation_evaluations as segmentation_evaluations
import niftynet.evaluation.regression_evaluations as regression_evaluations
from niftynet.evaluation.classification_evaluator import ClassificationEvaluator

from niftynet.evaluation.regression_evaluator import RegressionEvaluator

TEST_CASES = {0: {'seg_img': np.array([1, 0, 0, 0]), 'ref_img': np.array([1, 0, 0, 0])},
              1: {'seg_img': np.array([1, 0, 1, 0]), 'ref_img': np.array([1, 0, 0, 0])},
              2: {'seg_img': np.array([3, 2, 0, 0]), 'ref_img': np.array([1, 2, 0, 0])},
              3: {'seg_img': np.array([1, 0, 0.5, 0]), 'ref_img': np.array([1, 0, 0, 0])},
              4: {'seg_img': np.reshape([1, 1, 1, 0, 0, 0, 0, 0],[2,2,2,1,1]),
                  'ref_img': np.reshape([0, 0, 0, 0, 0, 0, 1, 1],[2,2,2,1,1])},

              }


class BinaryCheckTest(np.testing.TestCase):
    def test_binary_check_for_labels(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[2]['seg_img'],
                                             ref_img=TEST_CASES[2]['ref_img'])
        self.assertRaises(ValueError, pairwise_measures.check_binary)

    def test_binary_check_for_probabilities(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[3]['seg_img'],
                                             ref_img=TEST_CASES[3]['ref_img'])
        self.assertRaises(ValueError, pairwise_measures.check_binary)


class PairwiseTests(np.testing.TestCase):
    def test_dice_score(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[0]['seg_img'],
                                             ref_img=TEST_CASES[0]['ref_img'])
        self.assertEqual(pairwise_measures.dice_score(), 1.0)

    def test_true_positive(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.tp(), 1.0)

    def test_faulty_inputs(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[3]['seg_img'],
                                             ref_img=TEST_CASES[3]['ref_img'])
        self.assertRaises(ValueError, pairwise_measures.tp)

    def test_true_negative(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.tn(), 2.)

    def test_n_negative(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.n_neg_ref(), 3.)
        self.assertEqual(pairwise_measures.n_neg_seg(), 2.)

    def test_union(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.n_union(), 2.)

    def test_intersection(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.n_intersection(), 1.)

    def test_sensitivity(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.sensitivity(), 1.)

    def test_specificity(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.specificity(), 2. / 3)

    def test_accuracy(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.accuracy(), 3. / 4)

    def test_false_positive_rate(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.false_positive_rate(), 1. / 3)

    def test_positive_predictive_value(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.positive_predictive_values(), 1. / 2)

    def test_negative_predictive_value(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.negative_predictive_values(), 1.)

    def test_intersection_over_union(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.intersection_over_union(), 1. / 2)

    def test_jaccard(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.jaccard(), 1. / 2)

    def test_informedness(self):
        # true positive rate - false positive rate
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertAlmostEqual(pairwise_measures.informedness(), 2. / 3)

    def test_markedness(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'])
        self.assertEqual(pairwise_measures.markedness(), 1. / 2)

    def test_centre_of_mass(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'],
                                             pixdim=[2])
        self.assertListEqual(list(pairwise_measures.com_ref()), [0.0])
        self.assertListEqual(list(pairwise_measures.com_seg()), [1.0])
        self.assertEqual(pairwise_measures.com_dist(), 2.)

    def test_vol_diff(self):
        pairwise_measures = PairwiseMeasures(seg_img=TEST_CASES[1]['seg_img'],
                                             ref_img=TEST_CASES[1]['ref_img'],
                                             pixdim=[2])
        self.assertEqual(pairwise_measures.vol_diff(), 1.)

class MorphologyTests(np.testing.TestCase):
    def test_2d_offset(self):
        test_img = np.concatenate([np.zeros([3, 3]), np.ones([3, 3])])
        # expected border -- everywhere the test_img is 1, except the centre of it
        expected_border = np.zeros([6, 3])
        expected_border[3:][:] = 1
        expected_border[4, 1] = 0
        self.assertRaises(AssertionError, MorphologyOps, test_img, 8)
        #calculated_border = MorphologyOps(test_img, 8).border_map()
        #self.assertTrue(np.array_equal(calculated_border, expected_border))

    def test_3d_offset(self):
        test_img = np.zeros([10, 10, 10])
        test_img[5, 5, 5] = 1
        # border is the same as the test image -- just the one positive voxel
        calculated_border = MorphologyOps(test_img, 8).border_map()
        self.assertTrue(np.array_equal(test_img, calculated_border))

    def test_1d_error(self):
        test_img = np.zeros([1])
        self.assertRaises(AssertionError, MorphologyOps, test_img, 8)
        #self.assertRaises(ValueError, MorphologyOps(test_img, 8).border_map)


class RegressionEvaluationTests(np.testing.TestCase):
    def build_data(self):
        ref = np.reshape([1., .2, 2., 1., .9, .2, 3., 2.], [2, 2, 2, 1, 1])
        out = np.reshape([1., .3, 2., 1., .9, .2, 3., 2.], [2, 2, 2, 1, 1])
        return ref, out

    def test_mse(self):
        rd = regression_evaluations.mse(None, None, None).metric(
            *self.build_data())
        self.assertAlmostEqual(rd, 0.00125, 3)

    def test_rmse(self):
        rd = regression_evaluations.rmse(None, None, None).metric(
            *self.build_data())
        self.assertAlmostEqual(rd, 0.03535, 3)


    def test_mae(self):
        rd = regression_evaluations.mae(None, None, None).metric(
            *self.build_data())
        self.assertAlmostEqual(rd, 0.0125,  3)


class SegmentationEvaluationTests(np.testing.TestCase):
    def metric(self, cls, case):
        return cls(None, None, None).metric_from_binarized(
                                  seg=TEST_CASES[case]['seg_img'],
                                  ref=TEST_CASES[case]['ref_img'])

    def test_average_distance(self):
        self.assertAlmostEqual(self.metric(
            segmentation_evaluations.average_distance, 4), 1.2485,3)

    def test_hausdorff_distance(self):
        self.assertAlmostEqual(self.metric(
            segmentation_evaluations.hausdorff_distance, 4), 1.414,3)

    def test_hausdorff95_distance(self):
        self.assertAlmostEqual(self.metric(
            segmentation_evaluations.hausdorff95_distance, 4), 1.414,3)

    def test_dice_score(self):
        self.assertEqual(self.metric(segmentation_evaluations.dice, 0), 1.0)

    def test_true_positive(self):
        self.assertEqual(self.metric(segmentation_evaluations.tp, 1), 1.0)

    def test_true_negative(self):
        self.assertEqual(self.metric(segmentation_evaluations.tn, 1), 2.)

    def test_n_negative(self):
        self.assertEqual(self.metric(
            segmentation_evaluations.n_neg_ref, 1), 3.0)
        self.assertEqual(self.metric(
            segmentation_evaluations.n_neg_seg, 1), 2.0)

    def test_union(self):
        self.assertEqual(self.metric(segmentation_evaluations.n_union, 1), 2.0)

    def test_intersection(self):
        self.assertEqual(self.metric(
            segmentation_evaluations.n_intersection,1), 1.0)

    def test_sensitivity(self):
        self.assertEqual(self.metric(
            segmentation_evaluations.sensitivity, 1), 1.0)

    def test_specificity(self):
        self.assertEqual(self.metric(
            segmentation_evaluations.specificity, 1), 2. / 3)

    def test_accuracy(self):
        self.assertEqual(self.metric(
            segmentation_evaluations.accuracy, 1), 3. / 4)

    def test_false_positive_rate(self):
        self.assertEqual(self.metric(
            segmentation_evaluations.false_positive_rate, 1), 1. / 3)

    def test_positive_predictive_value(self):
        self.assertEqual(self.metric(
            segmentation_evaluations.positive_predictive_values, 1), 1. / 2)

    def test_negative_predictive_value(self):
        self.assertEqual(self.metric(
            segmentation_evaluations.negative_predictive_values, 1), 1.0)

    def test_intersection_over_union(self):
        self.assertEqual(self.metric(
            segmentation_evaluations.intersection_over_union, 1), 1. / 2)

    def test_jaccard(self):
        self.assertEqual(self.metric(
            segmentation_evaluations.jaccard, 1), 1. / 2)

    def test_informedness(self):
        self.assertAlmostEqual(self.metric(
            segmentation_evaluations.informedness, 1), 2. / 3)

    def test_markedness(self):
        self.assertEqual(self.metric(
            segmentation_evaluations.markedness,1), 1. /2)

class ClassificationEvaluationTests(np.testing.TestCase):
    def data1(self):
        raw_data = [[0,.12],[0,.24],[1,.36], [0,.45],
                    [0,.61],[1,.28],[1,.99], [1,.89]]
        formatted_data = [{'label':np.reshape(datum[0],[1,1,1,1,1]),
                           'inferred':np.reshape([1-datum[1],datum[1]],[1,1,1,1,2])} for datum in raw_data]
        return formatted_data
    def data2(self):
        raw_data = [[0,0],[0,0],[1,0],[0,0],
                    [0,1],[1,0],[1,1],[1,1]]
        formatted_data = [{'label':np.reshape(datum[0],[1,1,1,1,1]),
                           'inferred':np.reshape([datum[1]],[1,1,1,1,1])} for datum in raw_data]
        return formatted_data

    def generator(self, data):
        interp_orders = {'label':0,'inferred':-1}
        for idx, datum in enumerate(data):
            yield ('test'+str(idx), datum,interp_orders)

    def evaluator(self, eval_str, output_prob=True):
        class NS(object):
            def __init__(self, dict):
                self.__dict__.update(dict)
        classification_param=NS({'num_classes':2,
                                 'output_prob':output_prob})
        eval_param=NS({'evaluations':eval_str})
        return ClassificationEvaluator(None, classification_param, eval_param)

    def test_accuracy_output_prob(self):
        data = self.data1()
        evl = self.evaluator('niftynet.evaluation.classification_evaluations.accuracy')
        result_dict = evl.evaluate_from_generator(self.generator(data))
        self.assertIn((None,), result_dict)
        by_threshold = result_dict[(None,)].to_dict('index')
        
        self.assertEqual(by_threshold,
                      {0: {'accuracy': 0.625}})

    def test_accuracy_output_label(self):
        data = self.data2()
        evl = self.evaluator('niftynet.evaluation.classification_evaluations.accuracy', False)
        result_dict = evl.evaluate_from_generator(self.generator(data))
        self.assertIn((None,), result_dict)
        by_threshold = result_dict[(None,)].to_dict('index')
        
        self.assertEqual(by_threshold,
                      {0: {'accuracy': 0.625}})

    def test_contrib_roc(self):
        data = self.data1()
        evl = self.evaluator('niftynet.contrib.evaluation.classification_evaluations.roc')
        result_dict = evl.evaluate_from_generator(self.generator(data))
        self.assertIn(('thresholds',), result_dict)
        by_threshold = result_dict[('thresholds',)].to_dict('index')
        get_key = lambda x: [k for k in by_threshold.keys() if np.abs(k-x)<.01][0]
        sample = by_threshold[get_key(0.444)]
        self.assertEqual(sample['fp'],2)
        self.assertEqual(sample['spec'],0.5)
        self.assertEqual(sample['sens'],0.5)
        
#  FPF:   0.0000  0.0000  0.3333  0.3333  1.0000
#   TPF:   0.0000  0.6667  0.6667  1.0000  1.0000
#
#AREA UNDER ROC CURVE:
#  Area under fitted curve (Az) = 0.9043
#          Estimated std. error = 0.1260
#  Trapezoidal (Wilcoxon) area = 0.8889

    def test_contrib_roc_auc(self):
        data = self.data1()
        evl = self.evaluator('niftynet.contrib.evaluation.classification_evaluations.roc_auc')
        result_dict = evl.evaluate_from_generator(self.generator(data))
        self.assertIn((None,), result_dict)
        print(result_dict[(None,)].to_dict('index'))
        self.assertEqual(result_dict[(None,)].to_dict('index'),
                      {0: {'roc_auc': 0.71875}})


class SegmentationEvaluatorTests(np.testing.TestCase):
    """
    Tests that evaluator - evaluations integration works
    """

    class ReaderStub():
        def __init__(self):
            self.count = 0
            sz=[2,2,2,1,1]
            self.data=((0,
                        {'label': np.reshape([1, 0, 0, 0, 1, 0, 0, 0], sz),
                         'inferred': np.reshape([1, 0, 0, 0, 1, 0, 0, 0], sz)},
                        None),
                       (1,
                        {'label': np.reshape([1, 1, 0, 0, 1, 0, 0, 0], sz),
                         'inferred': np.reshape([1, 0, 0, 0, 1, 0, 0, 0], sz)},
                        None),
                       (-1, None, None))

        def __call__(self, shuffle):
            return_value = self.data[self.count]
            self.count += 1
            return return_value

        def get_subject_id(self, image_id):
            return ['foo','bar'][image_id]

    def test_segmentation_evaluator(self):
        app_param = Namespace(evaluation_units='label,cc',output_prob=False,
                              num_classes=2)
        eval_param = Namespace(evaluations='Dice,Jaccard,average_distance')
        evalu = SegmentationEvaluator(SegmentationEvaluatorTests.ReaderStub(),
                                      app_param, eval_param)
        result_dict = evalu.evaluate()
        self.assertIn(('subject_id', 'cc_id'), result_dict)
        self.assertIn(('subject_id', 'label'), result_dict)

        group_cc = result_dict[('subject_id', 'cc_id')]
        group_l = result_dict[('subject_id', 'label')]

        self.assertIn('jaccard', list(group_l.columns))
        self.assertIn('dice', list(group_l.columns))
        self.assertIn('jaccard', list(group_cc.columns))
        self.assertIn('dice', list(group_cc.columns))
        self.assertIn('average_distance', list(group_cc.columns))

        self.assertIn(('foo','r1_s1'), list(group_cc.index))
        self.assertIn(('bar','r1_s1'), list(group_cc.index))
        self.assertIn(('foo',1), list(group_l.index))
        self.assertIn(('bar',1), list(group_l.index))

class RegressionEvaluatorTests(np.testing.TestCase):
    """
    Tests that evaluator - evaluations integration works
    """

    class ReaderStub():
        def __init__(self):
            self.count = 0
            sz = [2, 2, 2, 1, 1]
            self.data = ((0,
                          {'output': np.reshape([1, 0, 0, 0, 1, 0, 0, 0],
                                               sz),
                           'inferred': np.reshape([1, 0, 0, 0, 1, 0, 0, 0],
                                                  sz)},
                          None),
                         (1,
                          {'output': np.reshape([1, 1, 0, 0, 1, 0, 0, 0],
                                               sz),
                           'inferred': np.reshape([1, 0, 0, 0, 1, 0, 0, 0],
                                                  sz)},
                          None),
                         (-1, None, None))

        def __call__(self, shuffle):
            return_value = self.data[self.count]
            self.count += 1
            return return_value

        def get_subject_id(self, image_id):
            return ['foo', 'bar'][image_id]

    def test_regression_evaluator(self):
        app_param = Namespace()
        eval_param = Namespace(evaluations='rmse,mse')
        evalu = RegressionEvaluator(RegressionEvaluatorTests.ReaderStub(),
                                    app_param, eval_param)
        result_dict = evalu.evaluate()
        self.assertIn(('subject_id',), result_dict)

        group = result_dict[('subject_id',)]
        self.assertEqual(('subject_id',), group.index.names)
        self.assertIn('mse', list(group.columns))
        self.assertIn('rmse', list(group.columns))
        self.assertIn('foo', list(group.index))
        self.assertIn('bar', list(group.index))

if __name__ == '__main__':
    tf.test.main()
