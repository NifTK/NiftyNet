# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.loss_classification_multi import LossFunction, variability
from tests.niftynet_testcase import NiftyNetTestCase

class VariabilityTests(NiftyNetTestCase):
    def test_variability_value(self):
        # test value is -0.5 * [1 * log(e / (1+e)) + 1 * log(e^2 / (e^2 + 1))]
        with self.cached_session():
            # [0,1,0] 2/3 , 1/3 4/9
            # [0,0,0] 1, 0 0
            predicted = [[0, 1,0],[1, 1,1]]

            computed_variability = variability(predicted, nrater=3)
            self.assertAlmostEqual(
                computed_variability[0].eval(),4.0/9.0)



class LossConfusionTest(NiftyNetTestCase):
    def test_confusion_matrix_loss(self):
        with self.cached_session():
            predicted = tf.constant([[[1,-1],[-1,1],[1,-1]],[[1,-1],[1,-1],[1,
                                                                       -1]]],
                                    dtype=tf.float32)
            predicted *= 1000
            ground_truth = [[0,0,1],[0,0,1]]
            test_loss_func = LossFunction(2, 3, loss_type='ConfusionMatrix',
                                          loss_func_params={'nrater':3})
            computed_loss = test_loss_func(ground_truth=ground_truth,
                                           pred_multi=predicted)
            self.assertAlmostEqual(computed_loss.eval(), 4.0/3.0)

class LossVariabilityTest(NiftyNetTestCase):
    def test_variability_loss(self):
        with self.cached_session():
            predicted = tf.constant([[[1,-1],[-1,1],[1,-1]],[[1,-1],[1,-1],[1,
                                                                       -1]]],
                                    dtype=tf.float32)
            predicted *= 1000
            ground_truth = [[0,0,1],[0,0,1]]
            test_loss_func = LossFunction(2, 3, loss_type='Variability')
            computed_loss = test_loss_func(ground_truth=ground_truth,
                                           pred_multi=predicted)
            self.assertAlmostEqual(computed_loss.eval(), np.sqrt(16.0/162.0))


class LossConsistencyTest(NiftyNetTestCase):
    def test_consistency_loss(self):
        with self.cached_session():
            predicted = tf.constant([[[1,-1],[-1,1],[1,-1]],[[1,-1],[1,-1],[1,
                                                                       -1]]],
                                    dtype=tf.float32)
            predicted *= 1000
            pred_ave = [[[0.66,0.33],[1,0]]]
            test_loss_func = LossFunction(2, 3, loss_type='Consistency')
            computed_loss = test_loss_func(pred_ave=pred_ave,
                                           pred_multi=predicted)
            self.assertAllClose(computed_loss.eval(), 0, atol=1E-2)



# class LossFunctionErrorTest(NiftyNetTestCase):
#     """
#     These tests check that a ValueError is called
#     for non-existent loss functions.
#     They also check that suggestions are returned
#     if the name is close to a real one.
#     """
#
#     def test_value_error_for_bad_loss_function(self):
#         with self.cached_session():
#             with self.assertRaises(ValueError):
#                 LossFunction(0, loss_type='wrong answer')
#
#     # Note: sensitive to precise wording of ValueError message.
#     def test_suggestion_for_dice_typo(self):
#         with self.cached_session():
#             with self.assertRaisesRegexp(ValueError, 'CrossEntropy'):
#                 LossFunction(0, loss_type='cross_entropy')


if __name__ == '__main__':
    tf.test.main()
