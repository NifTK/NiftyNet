import os

import numpy as np
import tensorflow as tf
from loss import dice, LossFunction, generalised_dice_loss, cross_entropy, sensitivity_specificity_loss

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


class SensitivitySpecificityTests(tf.test.TestCase):
    def test_sens_spec_loss_by_regression(self):
        with self.test_session():
            predicted = tf.constant([[0, 10], [10, 0], [10, 0], [10, 0]], dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            test_loss = sensitivity_specificity_loss(predicted, labels)
            self.assertAlmostEqual(test_loss.eval(), 2.06106e-9)


class GeneralisedDiceTest(tf.test.TestCase):
    def test_generalised_dice_score(self):
        with self.test_session():
            predicted = tf.constant([[0, 10], [10, 0], [10, 0], [10, 0]], dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            one_minus_generalised_dice_score = generalised_dice_loss(predicted, labels)
            print(one_minus_generalised_dice_score.eval())

    def test_gdsc_incorrect_type_weight_error(self):
        with self.test_session():
            with self.assertRaises(ValueError) as cm:
                predicted = tf.constant([[0, 10], [10, 0], [10, 0], [10, 0]], dtype=tf.float32, name='predicted')
                labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
                generalised_dice_loss(predicted, labels, type_weight='unknown')

            self.assertAllEqual(str(cm.exception),
                                'The variable type_weight "unknown" is not defined.')

    def test_generalised_dice_score_uniform(self):
        with self.test_session():
            predicted = tf.constant([[0, 10], [10, 0], [10, 0], [10, 0]], dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            one_minus_generalised_dice_score = generalised_dice_loss(predicted, labels, type_weight='Uniform')
            print(one_minus_generalised_dice_score.eval())


class DiceTest(tf.test.TestCase):
    def test_dice_score(self):
        with self.test_session():
            predicted = tf.constant([[0, 10], [10, 0], [10, 0], [10, 0]], dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            one_minus_dice_score = dice(predicted, labels)
            self.assertAllClose(one_minus_dice_score.eval(), 0.0, atol=1e-5)

    def test_wrong_prediction(self):
        with self.test_session():
            predicted = tf.constant([[0, 100]], dtype=tf.float32, name='predicted')
            labels = tf.constant([0], dtype=tf.int64, name='labels')
            one_minus_dice_score = dice(predicted, labels)
            self.assertAlmostEqual(one_minus_dice_score.eval(), 1.0)


class CrossEntropyTests(tf.test.TestCase):
    def test_cross_entropy_value(self):
        # test value is -0.5 * [1 * log(e / (1+e)) + 1 * log(e^2 / (e^2 + 1))]
        with self.test_session():
            predicted = tf.constant([[0, 1], [2, 0]], dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0], dtype=tf.int64, name='labels')
            computed_cross_entropy = cross_entropy(predicted, labels)
            self.assertAlmostEqual(computed_cross_entropy.eval(),
                                   -.5 * (np.log(np.e / (1 + np.e)) + np.log(np.e ** 2 / (1 + np.e ** 2))))


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
            with self.assertRaises(ValueError) as cm:
                LossFunction(0, loss_type='dice')
            self.assertAllEqual(str(cm.exception),
                                'By "dice", did you mean "Dice"?\n "dice" is not a valid loss.')

    def test_suggestion_for_gdsc_typo(self):
        with self.test_session():
            with self.assertRaises(ValueError) as cm:
                LossFunction(0, loss_type='GSDC')
            self.assertAllEqual(str(cm.exception),
                                'By "GSDC", did you mean "GDSC"?\n "GSDC" is not a valid loss.')


if __name__ == '__main__':
    tf.test.main()
