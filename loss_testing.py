import tensorflow as tf
from loss import dice


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

            self.assertAlmostEquals(one_minus_dice_score.eval(), 1.0)


if __name__ == '__main__':
    tf.test.main()
