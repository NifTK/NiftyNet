# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import tensorflow as tf
import numpy as np
from niftynet.evaluation.pairwise_measures import PairwiseMeasures
from niftynet.utilities.util_common import MorphologyOps

TEST_CASES = {0: {'seg_img': np.array([1, 0, 0, 0]), 'ref_img': np.array([1, 0, 0, 0])},
              1: {'seg_img': np.array([1, 0, 1, 0]), 'ref_img': np.array([1, 0, 0, 0])},
              2: {'seg_img': np.array([3, 2, 0, 0]), 'ref_img': np.array([1, 2, 0, 0])},
              3: {'seg_img': np.array([1, 0, 0.5, 0]), 'ref_img': np.array([1, 0, 0, 0])},
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


if __name__ == '__main__':
    tf.test.main()
