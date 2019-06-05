# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework.test_util import TensorFlowTestCase
from niftynet.engine.application_iteration import IterationMessage, \
    IterationMessageGenerator
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.signal import \
    TRAIN, ITER_FINISHED, GRAPH_CREATED, SESS_STARTED
from tests.application_driver_test import get_initialised_driver
from tests.niftynet_testcase import NiftyNetTestCase


class DriverLoopTest(TensorFlowTestCase):
    def test_interfaces(self):
        print("start test_interfaces")
        msg = IterationMessage()
        msg.current_iter = 0
        self.assertEqual(msg.current_iter, 0)
        print("assertEqual")
        self.assertEqual(msg.ops_to_run, {})
        print("assertEqual")
        self.assertEqual(msg.data_feed_dict, {})
        print("assertEqual")
        self.assertEqual(msg.current_iter_output, None)
        print("assertEqual")
        self.assertEqual(msg.should_stop, None)
        print("assertEqual")
        self.assertEqual(msg.phase, TRAIN)
        print("assertEqual")
        self.assertEqual(msg.is_training, True)
        print("assertEqual")
        self.assertEqual(msg.is_validation, False)
        print("assertEqual")
        self.assertEqual(msg.is_inference, False)
        print("assertEqual")
        msg.current_iter_output = {'test'}
        self.assertEqual(msg.current_iter_output, {'test'})
        print("assertEqual")
        self.assertGreater(msg.iter_duration, 0.0)
        print("assertGreater")
        self.assertStartsWith(msg.to_console_string(), 'training')
        print("assertStartsWith")
        self.assertEqual(msg.to_tf_summary(0), None)
        print("assertEqual")
        print("end test_interfaces")

    def test_set_fields(self):
        print("start test_set_fields")
        msg = IterationMessage()
        # setting iter will clear tic and iter output fields
        msg.current_iter = 3
        self.assertGreater(msg._current_iter_tic, 0.0)
        print("assertGreater")
        self.assertEqual(msg._current_iter_output, None)
        print("assertEqual")
        # setting iter output will update iter duration
        msg.current_iter_output = {CONSOLE: {'test': 'test'}}
        self.assertEqual(msg.current_iter, 3)
        print("assertEqual")
        self.assertGreater(msg.iter_duration, 0.0)
        print("assertGreater")
        self.assertRegexpMatches(msg.to_console_string(), '.*test=test.*')
        print("assertRegexpMatches")

        with self.assertRaisesRegexp(ValueError, ''):
            msg.current_iter = 'test'
        print("end test_set_fields")

    def test_run_vars(self):
        print("start test_run_vars")
        app_driver = get_initialised_driver()
        test_graph = app_driver.create_graph(app_driver.app, 1, True)
        test_tensor = test_graph.get_tensor_by_name(
            "G/conv_bn_selu/conv_/w:0")
        train_eval_msgs = []
        test_vals = []

        def get_iter_msgs(_sender, **msg):
            """" Captures iter_msg and model values for testing"""
            train_eval_msgs.append(msg['iter_msg'])
            test_vals.append(sess.run(test_tensor))
            print(msg['iter_msg'].to_console_string())

        ITER_FINISHED.connect(get_iter_msgs)

        with self.cached_session(graph=test_graph) as sess:
            SESS_STARTED.send(app_driver.app, iter_msg=None)
            iterations = IterationMessageGenerator(
                initial_iter=0,
                final_iter=3,
                validation_every_n=2,
                validation_max_iter=1,
                is_training_action=True)
            app_driver.loop(app_driver.app, iterations())

            # Check sequence of iterations
            self.assertRegexpMatches(
                train_eval_msgs[0].to_console_string(), 'training')
            print("assertRegexpMatches")
            self.assertRegexpMatches(
                train_eval_msgs[1].to_console_string(), 'training')
            print("assertRegexpMatches")
            self.assertRegexpMatches(
                train_eval_msgs[2].to_console_string(), 'validation')
            print("assertRegexpMatches")
            self.assertRegexpMatches(
                train_eval_msgs[3].to_console_string(), 'training')
            print("assertRegexpMatches")

            # Check durations
            for iter_msg in train_eval_msgs:
                self.assertGreater(iter_msg.iter_duration, 0.0)
                print("assertGreater")

            # Check training changes test tensor
            self.assertNotAlmostEqual(
                np.mean(np.abs(test_vals[0] - test_vals[1])), 0.0)
            print("assertNotAlmostEqual")

            self.assertNotAlmostEqual(
                np.mean(np.abs(test_vals[2] - test_vals[3])), 0.0)
            print("assertNotAlmostEqual")

            # Check validation doesn't change test tensor
            self.assertAlmostEqual(
                np.mean(np.abs(test_vals[1] - test_vals[2])), 0.0)
            print("assertAlmostEqual")

            app_driver.app.stop()

        ITER_FINISHED.disconnect(get_iter_msgs)
        print("end test_run_vars")


if __name__ == "__main__":
    tf.test.main()
