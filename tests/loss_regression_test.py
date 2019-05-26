# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.loss_regression import LossFunction
from niftynet.layer.loss_regression import l1_loss, l2_loss, huber_loss


class L1LossTests(tf.test.TestCase):
    def test_l1_loss_value(self):
        with self.test_session():
            predicted = tf.constant(
                [1, 1],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(loss_type='L1Loss')
            computed_l1_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(
                computed_l1_loss.eval(), 0.5)

    def test_l1_loss_value_weight(self):
        with self.test_session():
            weights = tf.constant(
                [[1, 2]], dtype=tf.float32, name='weights')
            predicted = tf.constant(
                [[1, 1]], dtype=tf.float32, name='predicted')
            labels = tf.constant([[1, 0]], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(loss_type='L1Loss')
            computed_l1_loss = test_loss_func(predicted, labels, weights)
            self.assertAlmostEqual(computed_l1_loss.eval(), 2.0 / 3.0)

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
    def test_l2_loss_value(self):
        with self.test_session():
            predicted = tf.constant(
                [[1, 2]], dtype=tf.float32, name='predicted')
            labels = tf.constant(
                [[1, 0]], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(loss_type='L2Loss')
            computed_l2_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(computed_l2_loss.eval(), 2.0)

    def test_l2_loss_value_weight(self):
        with self.test_session():
            weights = tf.constant(
                [[1, 2]], dtype=tf.float32, name='weights')
            predicted = tf.constant(
                [[1, 2]], dtype=tf.float32, name='predicted')
            labels = tf.constant(
                [[1, 0]], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(loss_type='L2Loss')
            computed_l2_loss = test_loss_func(predicted, labels, weights)
            self.assertAlmostEqual(
                computed_l2_loss.eval(), 8.0 / 9.0, places=3)

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


class HuberLossTests(tf.test.TestCase):
    def test_huber_loss(self):
        with self.test_session():
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            gold_standard = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='gold_standard')
            self.assertEqual(huber_loss(predicted, gold_standard).eval(), 0.0)

    def test_huber_continuous(self):
        with self.test_session():
            epsilon = tf.constant(
                1e-10, dtype=tf.float32)
            predicted = tf.constant(
                [1], dtype=tf.float32, name='predicted')
            gold_standard = tf.constant(
                [0], dtype=tf.float32, name='gold_standard')
            huber_loss_inside_delta = huber_loss(
                predicted + epsilon, gold_standard, delta=1.0)
            huber_loss_outside_delta = huber_loss(
                predicted - epsilon, gold_standard, delta=1.0)
            self.assertAlmostEqual(huber_loss_inside_delta.eval(),
                                   huber_loss_outside_delta.eval())

    def test_huber_loss_hand_example(self):
        with self.test_session():
            # loss should be: mean( 0.2 ** 2/ 2 + (2-0.5) ) == 1.52/2 == 0.76
            predicted = tf.constant(
                [1.2, 1], dtype=tf.float32, name='predicted')
            gold_standard = tf.constant(
                [1, 3], dtype=tf.float32, name='gold_standard')
            loss = huber_loss(predicted, gold_standard, delta=1.0)
            self.assertAlmostEqual(loss.eval(), .76)

    def test_huber_loss_value(self):
        with self.test_session():
            predicted = tf.constant(
                [[1, 2, 0.5]], dtype=tf.float32, name='predicted')
            labels = tf.constant(
                [[1, 0, 1]], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(loss_type='Huber')
            computed_huber_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(
                computed_huber_loss.eval(), 0.5417, places=4)

    def test_huber_loss_value_weight(self):
        with self.test_session():
            weights = tf.constant(
                [[1, 2, 1]], dtype=tf.float32, name='weights')
            predicted = tf.constant(
                [[1, 2, 0.5]], dtype=tf.float32, name='predicted')
            labels = tf.constant([[1, 0, 1]], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(loss_type='Huber')
            computed_huber_loss = test_loss_func(
                predicted, labels, weight_map=weights)
            self.assertAlmostEqual(
                computed_huber_loss.eval(), 3.125 / 4)


class RMSELossTests(tf.test.TestCase):
    def test_rmse_loss_value(self):
        with self.test_session():
            predicted = tf.constant(
                [[1.2, 1]], dtype=tf.float32, name='predicted')
            labels = tf.constant(
                [[1, 0]], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(loss_type='RMSE')
            computed_rmse_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(
                computed_rmse_loss.eval(), 0.7211, places=4)

    def test_rmse_loss_value_weight(self):
        with self.test_session():
            weights = tf.constant(
                [[1, 2.1]], dtype=tf.float32, name='weights')
            predicted = tf.constant(
                [[1, 1]], dtype=tf.float32, name='predicted')
            labels = tf.constant([[1, 0]], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(loss_type='RMSE')
            computed_rmse_loss = test_loss_func(predicted, labels, weights)
            self.assertAlmostEqual(
                computed_rmse_loss.eval(), 0.8231, places=4)


class MAELossTests(tf.test.TestCase):
    def test_MAE_loss_value(self):
        with self.test_session():
            predicted = tf.constant(
                [[1, 2]], dtype=tf.float32, name='predicted')
            labels = tf.constant(
                [[1.2, 0]], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(loss_type='MAE')
            computed_MAE_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(
                computed_MAE_loss.eval(), 1.1)

    def test_MAE_loss_value_weight(self):
        with self.test_session():
            weights = tf.constant(
                [[1, 2]], dtype=tf.float32, name='weights')
            predicted = tf.constant(
                [[1, 1]], dtype=tf.float32, name='predicted')
            labels = tf.constant([[1, 0]], dtype=tf.float32, name='labels')
            test_loss_func = LossFunction(loss_type='MAE')
            computed_MAE_loss = test_loss_func(predicted, labels, weights)
            self.assertAlmostEqual(
                computed_MAE_loss.eval(), 2.0 / 3.0)


if __name__ == '__main__':
    tf.test.main()
