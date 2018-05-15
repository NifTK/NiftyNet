# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.application_variables import CONSOLE
from niftynet.io.image_sets_partitioner import TRAIN, VALID
from tests.application_driver_test import get_initialised_driver


class RunVarsTest(tf.test.TestCase):
    def test_interfaces(self):
        msg = IterationMessage()
        msg.current_iter = 0
        self.assertEqual(msg.current_iter, 0)
        self.assertEqual(msg.ops_to_run, {})
        self.assertEqual(msg.data_feed_dict, {})
        self.assertEqual(msg.current_iter_output, None)
        self.assertEqual(msg.should_stop, False)
        self.assertEqual(msg.phase, TRAIN)
        self.assertEqual(msg.is_training, True)
        self.assertEqual(msg.is_validation, False)
        self.assertEqual(msg.is_inference, False)
        msg.current_iter_output = {'test'}
        self.assertEqual(msg.current_iter_output, {'test'})
        self.assertGreater(msg.iter_duration, 0.0)
        self.assertStartsWith(msg.to_console_string(), 'Training')
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

        with self.test_session(graph=test_graph) as sess:
            app_driver._run_sampler_threads(sess)
            sess.run(app_driver._init_op)

            iter_msg = IterationMessage()

            # run 1st training iter
            iter_msg.current_iter, iter_msg.phase = 1, TRAIN
            app_driver.run_vars(sess, iter_msg)
            model_value_1 = sess.run(test_tensor)
            self.assertGreater(iter_msg.iter_duration, 0.0)
            print(iter_msg.to_console_string())
            self.assertRegexpMatches(iter_msg.to_console_string(), 'Training')

            # run 2nd training iter
            iter_msg.current_iter, iter_msg.phase = 2, TRAIN
            app_driver.run_vars(sess, iter_msg)
            model_value_2 = sess.run(test_tensor)
            # make sure model gets updated
            self.assertNotAlmostEquals(
                np.mean(np.abs(model_value_1 - model_value_2)), 0.0)
            print(iter_msg.to_console_string())
            self.assertRegexpMatches(iter_msg.to_console_string(), 'Training')

            # run validation iter
            iter_msg.current_tier, iter_msg.phase = 3, VALID
            app_driver.run_vars(sess, iter_msg)
            model_value_3 = sess.run(test_tensor)
            # make sure model not gets udpated
            self.assertAlmostEquals(
                np.mean(np.abs(model_value_2 - model_value_3)), 0.0)
            print(iter_msg.to_console_string())
            self.assertRegexpMatches(iter_msg.to_console_string(), 'Validation')

            # run training iter
            iter_msg.current_iter, iter_msg.phase = 4, TRAIN
            app_driver.run_vars(sess, iter_msg)
            model_value_4 = sess.run(test_tensor)
            # make sure model gets updated
            self.assertNotAlmostEquals(
                np.mean(np.abs(model_value_2 - model_value_4)), 0.0)
            self.assertNotAlmostEquals(
                np.mean(np.abs(model_value_3 - model_value_4)), 0.0)
            print(iter_msg.to_console_string())
            self.assertRegexpMatches(iter_msg.to_console_string(), 'Training')

            app_driver.app.stop()
            self.assertEqual(iter_msg.ops_to_run, {})


if __name__ == "__main__":
    tf.test.main()
