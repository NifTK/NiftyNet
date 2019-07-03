# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.engine.application_iteration import IterationMessageGenerator
from niftynet.engine.application_variables import NETWORK_OUTPUT
from tests.application_driver_test import get_initialised_driver
from niftynet.engine.signal import SESS_STARTED
from tests.niftynet_testcase import NiftyNetTestCase


def set_iteration_update(msg):
    msg.ops_to_run[NETWORK_OUTPUT] = \
        tf.get_default_graph().get_tensor_by_name("G/conv_bn_selu/conv_/w:0")


class EventConsoleTest(NiftyNetTestCase):
    def create_interpreter(self):
        def mini_interpreter(np_array):
            self.assertEqual(np_array.shape, (10, 1, 20))
            return False

        return mini_interpreter

    def test_init(self):
        app_driver = get_initialised_driver()
        test_graph = app_driver.create_graph(app_driver.app, 1, True)

        app_driver.app.set_iteration_update = set_iteration_update
        app_driver.app.interpret_output = self.create_interpreter()

        app_driver.load_event_handlers(
            ['niftynet.engine.handler_model.ModelRestorer',
             'niftynet.engine.handler_network_output.OutputInterpreter',
             'niftynet.engine.handler_sampler.SamplerThreading'])
        with self.cached_session(graph=test_graph) as sess:
            SESS_STARTED.send(app_driver.app, iter_msg=None)

            iterator = IterationMessageGenerator(is_training_action=False)
            app_driver.loop(app_driver.app, iterator())
        app_driver.app.stop()


if __name__ == "__main__":
    tf.test.main()
