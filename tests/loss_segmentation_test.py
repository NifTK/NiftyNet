# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.loss_segmentation import LossFunction


class SensitivitySpecificityTests(tf.test.TestCase):
    # before re-factoring the code
    def test_sens_spec_loss_by_regression(self):
        with self.test_session():
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            test_loss_func = LossFunction(2, loss_type='SensSpec')
            test_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(test_loss.eval(), 2.06106e-9)

    def test_multi_label_sens_spec(self):
        with self.test_session():
            # answer calculated by hand -
            predicted = tf.constant(
                [[0, 1, 0], [0, 0, 1]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 2], dtype=tf.int64, name='labels')
            test_loss_func = LossFunction(3, loss_type='SensSpec',
                                          loss_func_params={'r': 0.05})
            test_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(test_loss.eval(), 0.14598623)


class GeneralisedDiceTest(tf.test.TestCase):
    # test done by regression for refactoring purposes
    def test_generalised_dice_score_regression(self):
        with self.test_session():
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            # weights = tf.cast(labels, tf.float32)
            test_loss_func = LossFunction(2, loss_type='GDSC')
            one_minus_generalised_dice_score = test_loss_func(
                predicted, labels)
            self.assertAllClose(
                one_minus_generalised_dice_score.eval(), 0.0, atol=1e-4)

    def test_gdsc_incorrect_type_weight_error(self):
        with self.test_session():
            with self.assertRaises(ValueError) as cm:
                predicted = tf.constant(
                    [[0, 10], [10, 0], [10, 0], [10, 0]],
                    dtype=tf.float32, name='predicted')
                labels = tf.constant(
                    [1, 0, 0, 0],
                    dtype=tf.int64, name='labels')
                test_loss_func = LossFunction(
                    2, loss_type='GDSC',
                    loss_func_params={'type_weight': 'unknown'})
                one_minus_generalised_dice_score = test_loss_func(predicted,
                                                                  labels)

    def test_generalised_dice_score_uniform_regression(self):
        with self.test_session():
            predicted = tf.constant([[0, 10], [10, 0], [10, 0], [10, 0]],
                                    dtype=tf.float32, name='predicted')

            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            weights = tf.cast(labels, tf.float32)
            test_loss_func = LossFunction(
                2, loss_type='GDSC',
                loss_func_params={'type_weight': 'Uniform'})
            one_minus_generalised_dice_score = test_loss_func(
                predicted, labels, weights)
            self.assertAllClose(one_minus_generalised_dice_score.eval(),
                                0.0, atol=1e-4)


class DiceTest(tf.test.TestCase):
    def test_dice_score(self):
        with self.test_session():
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            test_loss_func = LossFunction(2, loss_type='Dice')
            one_minus_dice_score = test_loss_func(predicted, labels)
            self.assertAllClose(one_minus_dice_score.eval(), 0.0, atol=1e-5)

    def test_dice_score_weights(self):
        with self.test_session():
            weights = tf.constant([1, 1, 0, 0], dtype=tf.float32,
                                  name='weights')
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            test_loss_func = LossFunction(2,
                                          loss_type='Dice')
            one_minus_dice_score = test_loss_func(predicted, labels,
                                                  weight_map=weights)
            self.assertAllClose(one_minus_dice_score.eval(), 0.0, atol=1e-4)

    def test_wrong_prediction(self):
        with self.test_session():
            predicted = tf.constant(
                [[0, 100]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([0], dtype=tf.int64, name='labels')
            test_loss_func = LossFunction(2, loss_type='Dice')
            one_minus_dice_score = test_loss_func(predicted, labels)
            self.assertAlmostEqual(one_minus_dice_score.eval(), 1.0)


class DiceTest_NS(tf.test.TestCase):
    def test_dice_score_nosquare(self):
        with self.test_session():
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            test_loss_func = LossFunction(2, loss_type='Dice_NS')
            one_minus_dice_score = test_loss_func(predicted, labels)
            self.assertAllClose(one_minus_dice_score.eval(), 0.0, atol=1e-4)

    def test_dice_score_nosquare_weights(self):
        with self.test_session():
            weights = tf.constant([1, 1, 0, 0], dtype=tf.float32,
                                  name='weights')
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            test_loss_func = LossFunction(2,
                                          loss_type='Dice_NS')
            one_minus_dice_score = test_loss_func(predicted, labels,
                                                  weight_map=weights)
            self.assertAllClose(one_minus_dice_score.eval(), 0.0, atol=1e-4)

    def test_wrong_prediction(self):
        with self.test_session():
            predicted = tf.constant(
                [[0, 100]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([0], dtype=tf.int64, name='labels')
            test_loss_func = LossFunction(2, loss_type='Dice_NS')
            one_minus_dice_score = test_loss_func(predicted, labels)
            self.assertAlmostEqual(one_minus_dice_score.eval(), 1.0)


class CrossEntropyTests(tf.test.TestCase):
    def test_cross_entropy_value(self):
        # test value is -0.5 * [1 * log(e / (1+e)) + 1 * log(e^2 / (e^2 + 1))]
        with self.test_session():
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

    def test_cross_entropy_value_weight(self):
        with self.test_session():
            weights = tf.constant([1, 2], dtype=tf.float32, name='weights')
            predicted = tf.constant(
                [[0, 1], [2, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0], dtype=tf.int64, name='labels')
            test_loss_func = LossFunction(2, loss_type='CrossEntropy')
            computed_cross_entropy = test_loss_func(predicted, labels, weights)
            self.assertAlmostEqual(
                computed_cross_entropy.eval(),
                -.5 * (
                    2.0 / 3.0 * np.log(np.e / (1 + np.e)) + 4.0 / 3.0 * np.log(
                        np.e ** 2 / (1 + np.e ** 2))))


class LossFunctionErrorsTest(tf.test.TestCase):
    """
    These tests check that a ValueError is called
    for non-existent loss functions.
    They also check that suggestions are returned
    if the name is close to a real one.
    """

    def test_value_error_for_bad_loss_function(self):
        with self.test_session():
            with self.assertRaises(ValueError):
                LossFunction(0, loss_type='wrong answer')

    # Note: sensitive to precise wording of ValueError message.
    def test_suggestion_for_dice_typo(self):
        with self.test_session():
            with self.assertRaisesRegexp(ValueError, 'Dice'):
                LossFunction(0, loss_type='dice')

    def test_suggestion_for_gdsc_typo(self):
        with self.test_session():
            with self.assertRaisesRegexp(ValueError, 'GDSC'):
                LossFunction(0, loss_type='GSDC')


if __name__ == '__main__':
    tf.test.main()
