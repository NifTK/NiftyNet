# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.loss_classification import LossFunction
from tests.niftynet_testcase import NiftyNetTestCase


class CrossEntropyTests(NiftyNetTestCase):
    def test_cross_entropy_value(self):
        # test value is -0.5 * [1 * log(e / (1+e)) + 1 * log(e^2 / (e^2 + 1))]
        with self.cached_session():
            predicted = tf.constant(
                [[0, 1], [2, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0], dtype=tf.int64, name='labels')
            test_loss_func = LossFunction(2, loss_type='CrossEntropy')
            computed_cross_entropy = test_loss_func(predicted, labels)
            self.assertAlmostEqual(
                computed_cross_entropy.eval(),
                -.5 * (np.log(np.e / (1 + np.e)) + np.log(
                    np.e ** 2 / (1 + np.e ** 2))))


class LossFunctionErrorsTest(NiftyNetTestCase):
    """
    These tests check that a ValueError is called
    for non-existent loss functions.
    They also check that suggestions are returned
    if the name is close to a real one.
    """

    def test_value_error_for_bad_loss_function(self):
        with self.cached_session():
            with self.assertRaises(ValueError):
                LossFunction(0, loss_type='wrong answer')

    # Note: sensitive to precise wording of ValueError message.
    def test_suggestion_for_dice_typo(self):
        with self.cached_session():
            with self.assertRaisesRegexp(ValueError, 'CrossEntropy'):
                LossFunction(0, loss_type='cross_entropy')


if __name__ == '__main__':
    tf.test.main()
