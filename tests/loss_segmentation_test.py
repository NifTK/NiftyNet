# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.layer.loss_segmentation import LossFunction
from niftynet.layer.loss_segmentation import huber_loss, l1_loss, l2_loss

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


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
                -.5 * (2.0/3.0 * np.log(np.e / (1 + np.e)) + 4.0/3.0 * np.log(
                    np.e ** 2 / (1 + np.e ** 2))))


class L1LossTests(tf.test.TestCase):
    def test_l1_loss_value(self):
        with self.test_session():
            predicted = tf.constant(
                [1, 1],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(1, loss_type='L1Loss')
            computed_l1_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(
                computed_l1_loss.eval(), 0.5)

    def test_l1_loss_value_weight(self):
        with self.test_session():
            weights = tf.constant([1, 2], dtype=tf.float32, name='weights')
            predicted = tf.constant(
                [1, 1],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(1, loss_type='L1Loss')
            computed_l1_loss = test_loss_func(predicted, labels, weights)
            self.assertAlmostEqual(
                computed_l1_loss.eval(), 2.0/3.0)


class L2LossTests(tf.test.TestCase):
    def test_l2_loss_value(self):
        with self.test_session():
            predicted = tf.constant(
                [1, 2],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(1, loss_type='L2Loss')
            computed_l2_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(
                computed_l2_loss.eval(), 4)

    def test_l2_loss_value_weight(self):
        with self.test_session():
            weights = tf.constant([1, 2], dtype=tf.float32, name='weights')
            predicted = tf.constant(
                [1, 2],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(1, loss_type='L2Loss')
            computed_l2_loss = test_loss_func(predicted, labels, weights)
            self.assertAlmostEqual(
                computed_l2_loss.eval(), 16.0 / 9.0)


class HuberLossTests(tf.test.TestCase):
    def test_huber_loss_value(self):
        with self.test_session():
            predicted = tf.constant(
                [1, 2, 0.5],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 1], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(1, loss_type='HuberLoss')
            computed_huber_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(
                computed_huber_loss.eval(), 1.675)

    def test_huber_loss_value_weight(self):
        with self.test_session():
            weights = tf.constant([1, 2, 1], dtype=tf.float32, name='weights')
            predicted = tf.constant(
                [1, 2, 0.5],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 1], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(1, loss_type='HuberLoss')
            computed_huber_loss = test_loss_func(predicted, labels, weights)
            self.assertAlmostEqual(
                computed_huber_loss.eval(), 3.125 / 4)


class LossFunctionErrorsTest(tf.test.TestCase):
    """
    These tests check that a ValueError is called for non-existent loss functions.
    They also check that suggestions are returned if the name is close to a real one.
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


class HuberLossTests(tf.test.TestCase):
    def test_huber_loss(self):
        with self.test_session():
            predicted = tf.constant([[0, 10], [10, 0], [10, 0], [10, 0]],
                                    dtype=tf.float32, name='predicted')
            gold_standard = tf.constant([[0, 10], [10, 0], [10, 0], [10, 0]],
                                        dtype=tf.float32, name='gold_standard')
            self.assertEqual(huber_loss(predicted, gold_standard).eval(), 0.0)

    def test_huber_continuous(self):
        with self.test_session():
            epsilon = tf.constant(1e-10, dtype=tf.float32)
            predicted = tf.constant([1], dtype=tf.float32, name='predicted')
            gold_standard = tf.constant([0], dtype=tf.float32,
                                        name='gold_standard')
            huber_loss_inside_delta = huber_loss(predicted + epsilon,
                                                 gold_standard, delta=1.0)
            huber_loss_outside_delta = huber_loss(predicted - epsilon,
                                                  gold_standard, delta=1.0)
            self.assertAlmostEqual(huber_loss_inside_delta.eval(),
                                   huber_loss_outside_delta.eval())

    def test_huber_loss_hand_example(self):
        with self.test_session():
            # loss should be: mean( 0.2 ** 2/ 2 + (2-0.5) ) == 1.52/2 == 0.76
            predicted = tf.constant([1.2, 1], dtype=tf.float32,
                                    name='predicted')
            gold_standard = tf.constant([1, 3], dtype=tf.float32,
                                        name='gold_standard')
            loss = huber_loss(predicted, gold_standard, delta=1.0)
            self.assertAlmostEqual(loss.eval(), .76)


class L1LossTests(tf.test.TestCase):
    def test_l1_loss(self):
        # expected loss: mean(.2 + 2) = 1.1
        with self.test_session():
            predicted = tf.constant(
                [1.2, 1],
                dtype=tf.float32, name='predicted')
            gold_standard = tf.constant(
                [1, 3], dtype=tf.float32, name='gold_standard')
            self.assertAlmostEqual(
                l1_loss(predicted, gold_standard).eval(), 1.1)


class L2LossTests(tf.test.TestCase):
    def test_l2_loss(self):
        # expected loss: (0.04 + 4 + 1) /2 = 2.52
        # (note - not the mean, just the sum)
        with self.test_session():
            predicted = tf.constant(
                [1.2, 1, 2],
                dtype=tf.float32, name='predicted')
            gold_standard = tf.constant(
                [1, 3, 3],
                dtype=tf.float32, name='gold_standard')
            self.assertAlmostEqual(
                l2_loss(predicted, gold_standard).eval(), 2.52)


if __name__ == '__main__':
    tf.test.main()
