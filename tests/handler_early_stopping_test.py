# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.engine.handler_early_stopping import check_should_stop


class EarlyStopperTest(tf.test.TestCase):

    def test_mean(self):
        should_stop = check_should_stop(mode='mean',
                                        performance_history=[1, 2, 1, 2, 1, 2, 1, 2, 3],
                                        patience=3)
        self.assertTrue(should_stop)
        should_stop = check_should_stop(mode='mean',
                                        performance_history=[1, 2, 1, 2, 1, 2, 1, 2, 3, 0],
                                        patience=3)
        self.assertFalse(should_stop)

    def test_robust_mean(self):
        should_stop = check_should_stop(mode='robust_mean',
                                        performance_history=[1, 2, 1, 2, 1, 2, 1, 200, -100, 1.2],
                                        patience=6)
        self.assertFalse(should_stop)
        should_stop = check_should_stop(mode='robust_mean',
                                        performance_history=[1, 2, 1, 2, 1, 2, 1, 200, -100, 1.4],
                                        patience=6)
        self.assertTrue(should_stop)

    def test_median(self):
        should_stop = check_should_stop(mode='median',
                                        performance_history=[1, 2, 1, 2, 1, 2, 1, 2, 3],
                                        patience=3)
        self.assertTrue(should_stop)
        should_stop = check_should_stop(mode='median',
                                        performance_history=[1, 2, 1, 2, 1, 2, 1, 2, 3, 0],
                                        patience=3)
        self.assertFalse(should_stop)

    def test_generalisation_loss(self):
        should_stop = check_should_stop(mode='generalisation_loss',
                                        performance_history=[1, 2, 1, 2, 1, 2, 1, 2, 3],
                                        patience=6)
        self.assertTrue(should_stop)
        should_stop = check_should_stop(mode='generalisation_loss',
                                        performance_history=[1, 2, 1, 2, 3, 2, 1, 2, 1],
                                        patience=6)
        self.assertFalse(should_stop)

    def test_robust_median(self):
        should_stop = check_should_stop(mode='robust_median',
                                        performance_history=[1, 2, 1, 2, 1, 2, 1, 200, -100, 0.9],
                                        patience=6)
        self.assertFalse(should_stop)
        should_stop = check_should_stop(mode='robust_median',
                                        performance_history=[1, 2, 1, 2, 1, 2, 1, 200, -100, 1.1],
                                        patience=6)
        self.assertTrue(should_stop)

    def test_median_smoothing(self):
        should_stop = check_should_stop(mode='median_smoothing',
                                        performance_history=get_data(),
                                        patience=8)
        self.assertTrue(should_stop)

    def test_weird_mode(self):
        check_should_stop(mode='adslhfjdkas', performance_history=get_data(), patience=3)
        self.assertRaises(Exception)

    def test_no_hist(self):
        should_stop = check_should_stop(mode='mean', performance_history=[], patience=3)
        self.assertFalse(should_stop)


if __name__ == "__main__":
    tf.test.main()
