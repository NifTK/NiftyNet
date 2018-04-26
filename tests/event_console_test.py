# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from tests.application_driver_test import get_initialised_driver
from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.application_variables import global_vars_init_or_restore
#from niftynet.engine.signal import SESS_FINISHED, SESS_STARTED, ITER_FINISHED
from niftynet.engine.signal import ITER_FINISHED


class EventConsoleTest(tf.test.TestCase):
    def test_init(self):
        ITER_FINISHED.connect(self.iteration_listener)

        app_driver = get_initialised_driver()
        test_graph = app_driver.create_graph(app_driver.graph)
        app_driver.load_event_handlers(
            ['niftynet.engine.event_sampler.SamplerThreading',
             'niftynet.engine.event_console.ConsoleLogger'])
        with self.test_session(graph=test_graph) as sess:
            sess.run(global_vars_init_or_restore())
            msg = IterationMessage()
            app_driver.loop([msg])

    def iteration_listener(self, sender, **msg):
        msg = msg['iter_msg']
        self.assertRegexpMatches(msg.to_console_string(), 'mean')
        self.assertRegexpMatches(msg.to_console_string(), 'var')


if __name__ == "__main__":
    tf.test.main()
