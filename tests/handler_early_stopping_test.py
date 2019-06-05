# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from tests.application_driver_test import get_initialised_driver
from niftynet.engine.application_iteration import IterationMessage
from niftynet.engine.signal import SESS_STARTED, ITER_FINISHED, VALID


class EarlyStopperTest(tf.test.TestCase):
    def test_init(self):
        ITER_FINISHED.connect(self.iteration_listener)
        app_driver = get_initialised_driver()
        app_driver.load_event_handlers(
            ['niftynet.engine.handler_model.ModelRestorer',
             'niftynet.engine.handler_console.ConsoleLogger',
             'niftynet.engine.handler_sampler.SamplerThreading',
             'niftynet.engine.handler_performance.PerformanceLogger',
             'niftynet.engine.handler_early_stopping.EarlyStopper'])
        graph = app_driver.create_graph(app_driver.app, 1, True)
        with self.test_session(graph=graph) as sess:
            for i in range(250):
                SESS_STARTED.send(app_driver.app, iter_msg=None)
                msg = IterationMessage()
                msg._phase = VALID
                msg.current_iter = i
                app_driver.loop(app_driver.app, [msg])
        app_driver.app.stop()
        ITER_FINISHED.disconnect(self.iteration_listener)

    def iteration_listener(self, sender, **msg):
        msg = msg['iter_msg']
        self.assertRegexpMatches(msg.to_console_string(), 'total_loss')
        if len(sender.performance_history) >= sender.patience:

            self.assertTrue()



if __name__ == "__main__":
    tf.test.main()
