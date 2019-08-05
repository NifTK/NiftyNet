# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

from tests.application_driver_test import get_initialised_driver
from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.signal import SESS_STARTED, ITER_FINISHED, VALID
from tests.niftynet_testcase import NiftyNetTestCase


class PerformanceLoggerTest(NiftyNetTestCase):
    def test_init(self):
        ITER_FINISHED.connect(self.iteration_listener)
        app_driver = get_initialised_driver()
        app_driver.load_event_handlers(
            ['niftynet.engine.handler_model.ModelRestorer',
             'niftynet.engine.handler_console.ConsoleLogger',
             'niftynet.engine.handler_sampler.SamplerThreading',
             'niftynet.engine.handler_performance.PerformanceLogger'])
        graph = app_driver.create_graph(app_driver.app, 1, True)
        with self.cached_session(graph=graph) as sess:
            for i in range(110):
                SESS_STARTED.send(app_driver.app, iter_msg=None)
                msg = IterationMessage()
                msg._phase = VALID
                msg.current_iter = i
                app_driver.loop(app_driver.app, [msg])
        app_driver.app.stop()
        ITER_FINISHED.disconnect(self.iteration_listener)

    def iteration_listener(self, sender, **msg):
        msg = msg['iter_msg']
        self.assertRegexpMatches(msg.to_console_string(), '.*total_loss.*')
        if msg.current_iter > 1:
            self.assertTrue(isinstance(sender.performance_history, list))
            self.assertTrue(len(sender.performance_history) <= sender.patience)
            self.assertTrue(all([isinstance(p, np.float32) for p in sender.performance_history]))


if __name__ == "__main__":
    tf.test.main()
