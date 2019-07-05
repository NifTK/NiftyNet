# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from tests.application_driver_test import get_initialised_driver
from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.signal import SESS_STARTED, ITER_FINISHED
from tests.niftynet_testcase import NiftyNetTestCase


class EventConsoleTest(NiftyNetTestCase):
    def test_init(self):
        ITER_FINISHED.connect(self.iteration_listener)

        app_driver = get_initialised_driver()
        app_driver.load_event_handlers(
            ['niftynet.engine.handler_model.ModelRestorer',
             'niftynet.engine.handler_console.ConsoleLogger',
             'niftynet.engine.handler_sampler.SamplerThreading'])
        graph = app_driver.create_graph(app_driver.app, 1, True)
        with self.cached_session(graph=graph) as sess:
            SESS_STARTED.send(app_driver.app, iter_msg=None)
            msg = IterationMessage()
            msg.current_iter = 1
            app_driver.loop(app_driver.app, [msg])
        app_driver.app.stop()

        ITER_FINISHED.disconnect(self.iteration_listener)

    def iteration_listener(self, sender, **msg):
        msg = msg['iter_msg']
        self.assertRegexpMatches(msg.to_console_string(), 'mean')
        self.assertRegexpMatches(msg.to_console_string(), 'var')


if __name__ == "__main__":
    tf.test.main()
