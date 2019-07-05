# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

from niftynet.engine.handler_early_stopping import check_should_stop
from tests.niftynet_testcase import NiftyNetTestCase

class EarlyStopperTest(NiftyNetTestCase):

    def test_mean(self):
        should_stop = check_should_stop(mode='mean',
                                        performance_history=[1, 2, 1, 2, 1,
                                                             2, 1, 2, 3])
        self.assertTrue(should_stop)
        should_stop = check_should_stop(mode='mean',
                                        performance_history=[1, 2, 1, 2, 1, 2,
                                                             1, 2, 3, 0])
        self.assertFalse(should_stop)

    def test_robust_mean(self):
        should_stop = check_should_stop(mode='robust_mean',
                                        performance_history=[1, 2, 1, 2, 1, 2,
                                                             1, 200, -10, 1.4])
        self.assertFalse(should_stop)
        should_stop = check_should_stop(mode='robust_mean',
                                        performance_history=[1, 2, 1, 2, 1, 2,
                                                             1, 200, -10, 1.5])
        self.assertTrue(should_stop)

    def test_median(self):
        should_stop = check_should_stop(mode='median',
                                        performance_history=[1, 2, 1, 2, 1, 2,
                                                             1, 2, 3])
        self.assertTrue(should_stop)
        should_stop = check_should_stop(mode='median',
                                        performance_history=[1, 2, 1, 2, 1, 2,
                                                             1, 2, 3, 0])
        self.assertFalse(should_stop)

    def test_generalisation_loss(self):
        should_stop = check_should_stop(mode='generalisation_loss',
                                        performance_history=[1, 2, 1, 2, 1,
                                                             2, 1, 2, 3])
        self.assertTrue(should_stop)
        should_stop = check_should_stop(mode='generalisation_loss',
                                        performance_history=[1, 2, 1, 2, 3,
                                                             2, 1, 2, 1])
        self.assertFalse(should_stop)

    def test_validation_up(self):
        data = []
        for i in range(10):
            data.extend(np.arange(1, 9))
            data.extend(np.arange(2, 10)[::-1])
        should_stop = check_should_stop(mode='validation_up',
                                        performance_history=
                                        np.arange(0, 20) / 10.0)
        print("1 val")
        self.assertTrue(should_stop)
        should_stop = check_should_stop(mode='validation_up',
                                        performance_history=np.arange(
                                            0, 20)[::-1] / 10)
        print("2 val")
        self.assertFalse(should_stop)

        should_stop = check_should_stop(mode='validation_up',
                                        performance_history=data,
                                        min_delta=0.2)
        print("3 val")
        self.assertFalse(should_stop)

    def test_median_smoothing(self):
        data = []
        for i in range(10):
            data.extend(np.arange(0, 8))
            data.extend(np.arange(1, 9)[::-1])
        should_stop = \
            check_should_stop(mode='median_smoothing',
                              performance_history=np.arange(0, 20) / 10.0)
        self.assertTrue(should_stop)
        should_stop = check_should_stop(mode='median_smoothing',
                                        performance_history=np.arange(
                                            0, 20)[::-1] / 10)
        self.assertFalse(should_stop)

        should_stop = check_should_stop(mode='median_smoothing',
                                        performance_history=data)
        self.assertFalse(should_stop)

    def test_weird_mode(self):
        with self.assertRaises(Exception):
            check_should_stop(mode='adslhfjdkas',
                              performance_history=[1, 2, 3, 4, 5, 6, 7, 8, 9])


if __name__ == "__main__":
    tf.test.main()
