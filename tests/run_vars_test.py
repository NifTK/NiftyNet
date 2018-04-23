# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.signal import TRAIN, ITER_FINISHED
from niftynet.engine.application_driver import train_iter_generator
from tests.application_driver_test import get_initialised_driver


class DriverLoopTest(tf.test.TestCase):
    def test_interfaces(self):
        msg = IterationMessage()
        msg.current_iter = 0
        self.assertEqual(msg.current_iter, 0)
        self.assertEqual(msg.ops_to_run, {})
        self.assertEqual(msg.data_feed_dict, {})
        self.assertEqual(msg.current_iter_output, None)
        self.assertEqual(msg.should_stop, None)
        self.assertEqual(msg.phase, TRAIN)
        self.assertEqual(msg.is_training, True)
        self.assertEqual(msg.is_validation, False)
        self.assertEqual(msg.is_inference, False)
        msg.current_iter_output = {'test'}
        self.assertEqual(msg.current_iter_output, {'test'})
        self.assertGreater(msg.iter_duration, 0.0)
        self.assertStartsWith(msg.to_console_string(), 'training')
        self.assertEqual(msg.to_tf_summary(0), None)

    def test_set_fields(self):
        msg = IterationMessage()

        # setting iter will clear tic and iter output fields
        msg.current_iter = 3
        self.assertGreater(msg._current_iter_tic, 0.0)
        self.assertEqual(msg._current_iter_output, None)

        # setting iter output will update iter duration
        msg.current_iter_output = {CONSOLE: {'test': 'test'}}
        self.assertEqual(msg.current_iter, 3)
        self.assertGreater(msg.iter_duration, 0.0)
        self.assertRegexpMatches(msg.to_console_string(), '.*test=test.*')

        with self.assertRaisesRegexp(ValueError, ''):
            msg.current_iter = 'test'

    def test_run_vars(self):
        app_driver = get_initialised_driver()
        test_graph = app_driver._create_graph(app_driver.graph)
        test_tensor = app_driver.graph.get_tensor_by_name(
            "G/conv_bn_selu/conv_/w:0")
        app_driver.load_event_handlers(
            ['niftynet.engine.event_gradient.ApplyGradients'])

        iter_msgs = []
        test_vals = []

        def get_iter_msgs(_sender, **msg):
            """" Captures iter_msg and model values for testing"""
            iter_msg = msg['iter_msg']
            iter_msgs.append(iter_msg)
            test_vals.append(sess.run(test_tensor))
            print(iter_msg.to_console_string())
        ITER_FINISHED.connect(get_iter_msgs)

        app_driver.initial_iter=0
        app_driver.final_iter=3
        app_driver.validation_every_n = 2
        app_driver.validation_max_iter = 1
        loop_status={}

        with self.test_session(graph=test_graph) as sess:
            app_driver._run_sampler_threads()
            app_driver._run_sampler_threads(sess)
            sess.run(app_driver._init_op)

            iterations = train_iter_generator(
                    app_driver.initial_iter,
                    app_driver.final_iter,
                    app_driver.validation_every_n,
                    app_driver.validation_max_iter)
            app_driver._loop(iterations, sess, loop_status)

            # Check sequence of iterations
            self.assertRegexpMatches(
                iter_msgs[0].to_console_string(), 'training')
            self.assertRegexpMatches(
                iter_msgs[1].to_console_string(), 'training')
            self.assertRegexpMatches(
                iter_msgs[2].to_console_string(), 'validation')
            self.assertRegexpMatches(
                iter_msgs[3].to_console_string(), 'training')

            # Check durations
            for iter_msg in iter_msgs:
                self.assertGreater(iter_msg.iter_duration, 0.0)

            # Check training changes test tensor
            self.assertNotAlmostEqual(
                np.mean(np.abs(test_vals[0] - test_vals[1])), 0.0)
            self.assertNotAlmostEqual(
                np.mean(np.abs(test_vals[2] - test_vals[3])), 0.0)

            # Check validation doesn't change test tensor
            self.assertAlmostEqual(
                np.mean(np.abs(test_vals[1] - test_vals[2])), 0.0)

            app_driver.app.stop()


if __name__ == "__main__":
    tf.test.main()
