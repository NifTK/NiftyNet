import tensorflow as tf
import os
from loss import dice, GDSC_loss, LossFunction
import sys
from io import StringIO

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


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


# TODO implement tests of other loss functions
# class GeneralisedDiceTest(tf.test.TestCase):
#     def test_generalised_dice_score(self):
#         with self.test_session():
#             predicted = tf.constant([[0, 10], [10, 0], [10, 0], [10, 0]], dtype=tf.float32, name='predicted')
#             labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
#             one_minus_generalised_dice_score = GDSC_loss(predicted, labels)
#             print(one_minus_generalised_dice_score.eval())


class ErrorThrowingTest(tf.test.TestCase):
    """
    These tests check that a ValueError is called for non-existent loss functions. 
    It also checks that a suggestion is returned if the name is close to a real one.
    """

    def test_value_error_for_bad_loss_function(self):
        with self.test_session():
            with self.assertRaises(ValueError):
                LossFunction(0, loss_type='wrong answer')

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
