# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import uuid
import numpy as np
import tensorflow as tf

from niftynet.engine.application_driver import ApplicationDriver
from niftynet.engine.application_variables import global_vars_init_or_restore
from niftynet.engine.handler_model import ModelRestorer
from niftynet.io.misc_io import set_logger
from niftynet.utilities.util_common import ParserNamespace
from niftynet.engine.signal import SESS_FINISHED, SESS_STARTED
from tests.niftynet_testcase import NiftyNetTestCase

# def _run_test_application():
#    test_driver = get_initialised_driver()
#    test_driver.run_application()
#    return


def get_initialised_driver(starting_iter=0,
                           model_dir_rand=True,
                           vars_to_restore='',
                           application='tests.toy_application.ToyApplication'):
    if model_dir_rand:
        model_dir = os.path.join('.', 'testing_data', 'tmp', str(uuid.uuid4()))
        os.makedirs(model_dir)
    else:
        model_dir = os.path.join('.', 'testing_data')
    system_param = {
        'SYSTEM': ParserNamespace(
            action='train',
            num_threads=2,
            num_gpus=4,
            cuda_devices='6',
            model_dir=model_dir,
            dataset_split_file=os.path.join(
                '.', 'testing_data', 'testtoyapp.csv'),
            event_handler=[
                'niftynet.engine.handler_model.ModelRestorer',
                'niftynet.engine.handler_sampler.SamplerThreading',
                'niftynet.engine.handler_gradient.ApplyGradients'],
            iteration_generator=None),
        'NETWORK': ParserNamespace(
            batch_size=20,
            name='tests.toy_application.TinyNet'),
        'TRAINING': ParserNamespace(
            starting_iter=starting_iter,
            max_iter=500,
            save_every_n=20,
            tensorboard_every_n=1,
            max_checkpoints=20,
            optimiser='niftynet.engine.application_optimiser.Adagrad',
            validation_every_n=-1,
            exclude_fraction_for_validation=0.1,
            exclude_fraction_for_inference=0.1,
            vars_to_restore=vars_to_restore,
            patience=100,
            lr=0.01),
        'CUSTOM': ParserNamespace(
            vector_size=100,
            mean=10.0,
            stddev=2.0,
            name=application)
    }
    app_driver = ApplicationDriver()
    app_driver.initialise_application(system_param, {})
    # set parameters without __init__
    app_driver.app.action_param = system_param['TRAINING']
    app_driver.app.net_param = system_param['NETWORK']
    app_driver.app.action = 'train'
    return app_driver


class ApplicationDriverTest(NiftyNetTestCase):
    def test_wrong_init(self):
        app_driver = ApplicationDriver()
        with self.assertRaisesRegexp(AttributeError, ''):
            app_driver.initialise_application([], [])

    # def test_create_app(self):
    #     test_driver = get_initialised_driver(499, True)
    #     with self.assertRaisesRegexp(ValueError, 'Could not import'):
    #         test_driver._create_app('test.test')
    #     with self.assertRaisesRegexp(ValueError, 'Could not import'):
    #         test_driver._create_app('testtest')
    #     with self.assertRaisesRegexp(ValueError, 'Could not import'):
    #         test_driver._create_app(1)
    #     test_driver._create_app('tests.toy_application.ToyApplication')

    # def test_stop_app(self):
    #     test_driver = get_initialised_driver()
    #     graph = test_driver.create_graph(
    #         test_driver.app, test_driver.num_gpus, True)
    #     with self.cached_session(graph=graph) as sess:
    #         sess.run(global_vars_init_or_restore())
    #         GRAPH_CREATED.send(test_driver.app, iter_msg=None)
    #         SESS_STARTED.send(test_driver.app, iter_msg=None)
    #         train_op = test_driver.app.gradient_op
    #         SESS_FINISHED.send(test_driver.app, itermsg=None)
    #         test_driver.app.stop()
    #         try:
    #             while True:
    #                 sess.run(train_op)
    #         except tf.errors.OutOfRangeError:
    #             for thread in test_driver.app.sampler[0][0]._threads:
    #                 self.assertFalse(thread.isAlive(), "threads not closed")

    def test_training_update(self):
        test_driver = get_initialised_driver()
        graph = test_driver.create_graph(test_driver.app, 1, True)
        with self.cached_session(graph=graph) as sess:
            SESS_STARTED.send(test_driver.app, iter_msg=None)

            train_op = test_driver.app.gradient_op
            test_tensor = graph.get_tensor_by_name(
                'G/conv_bn_selu/conv_/w:0')
            var_0 = sess.run(test_tensor)
            sess.run(train_op)
            var_1 = sess.run(test_tensor)
            square_diff = np.sum(np.abs(var_0 - var_1))
            self.assertGreater(
                square_diff, 0.0, 'train_op does not change model')
            SESS_FINISHED.send(test_driver.app, itermsg=None)
            test_driver.app.stop()

    def test_multi_device_inputs(self):
        test_driver = get_initialised_driver()
        graph = test_driver.create_graph(
            test_driver.app, test_driver.num_gpus, True)
        with self.cached_session(graph=graph) as sess:
            SESS_STARTED.send(test_driver.app, iter_msg=None)
            for i in range(2):
                sess.run(test_driver.app.gradient_op)
                s_0, s_1, s_2, s_3 = sess.run([
                    graph.get_tensor_by_name(
                        'worker_0/feature_input:0'),
                    graph.get_tensor_by_name(
                        'worker_1/feature_input:0'),
                    graph.get_tensor_by_name(
                        'worker_2/feature_input:0'),
                    graph.get_tensor_by_name(
                        'worker_3/feature_input:0')
                ])
                msg = 'same input data for different devices'
                self.assertGreater(np.sum(np.abs(s_0 - s_1)), 0.0, msg)
                self.assertGreater(np.sum(np.abs(s_0 - s_2)), 0.0, msg)
                self.assertGreater(np.sum(np.abs(s_0 - s_3)), 0.0, msg)
                self.assertGreater(np.sum(np.abs(s_1 - s_2)), 0.0, msg)
                self.assertGreater(np.sum(np.abs(s_1 - s_3)), 0.0, msg)
                self.assertGreater(np.sum(np.abs(s_2 - s_3)), 0.0, msg)
            SESS_FINISHED.send(test_driver.app, itermsg=None)
            test_driver.app.stop()

    def test_multi_device_gradients(self):
        test_driver = get_initialised_driver()
        graph = test_driver.create_graph(
            test_driver.app, test_driver.num_gpus, True)
        with self.cached_session(graph=graph) as sess:
            SESS_STARTED.send(test_driver.app, iter_msg=None)
            for i in range(2):
                sess.run(test_driver.app.gradient_op)
                g_0, g_1, g_2, g_3, g_ave = sess.run([
                    graph.get_tensor_by_name(
                        'worker_0/ComputeGradients/gradients/AddN_5:0'),
                    graph.get_tensor_by_name(
                        'worker_1/ComputeGradients/gradients/AddN_5:0'),
                    graph.get_tensor_by_name(
                        'worker_2/ComputeGradients/gradients/AddN_5:0'),
                    graph.get_tensor_by_name(
                        'worker_3/ComputeGradients/gradients/AddN_5:0'),
                    graph.get_tensor_by_name(
                        'ApplyGradients/AveOverDevices:0')
                ])
                self.check_gradients(g_0, g_1, g_2, g_3, g_ave)
            SESS_FINISHED.send(test_driver.app, itermsg=None)
            test_driver.app.stop()

    def test_multi_device_multi_optimiser_gradients(self):
        test_driver = get_initialised_driver(
            application='tests.toy_application.ToyApplicationMultOpti')
        graph = test_driver.create_graph(
            test_driver.app, test_driver.num_gpus, True)
        with self.cached_session(graph=graph) as sess:
            SESS_STARTED.send(test_driver.app, iter_msg=None)
            for i in range(2):
                sess.run(test_driver.app.gradient_op)
                # query generator gradient sample to check
                dis_0, dis_1, dis_2, dis_3, dis_ave = sess.run([
                    graph.get_tensor_by_name(
                        'worker_0/ComputeGradientsD/gradients/AddN_5:0'),
                    graph.get_tensor_by_name(
                        'worker_1/ComputeGradientsD/gradients/AddN_5:0'),
                    graph.get_tensor_by_name(
                        'worker_2/ComputeGradientsD/gradients/AddN_5:0'),
                    graph.get_tensor_by_name(
                        'worker_3/ComputeGradientsD/gradients/AddN_5:0'),
                    graph.get_tensor_by_name(
                        'ApplyGradients/AveOverDevices:0')
                ])

                # query discriminator gradient sample to check
                gen_0, gen_1, gen_2, gen_3, gen_ave = sess.run([
                    graph.get_tensor_by_name(
                        'worker_0/ComputeGradientsG/gradients/worker_0/tinynet/G/conv/conv_/conv/ExpandDims_1_grad/Reshape:0'),
                    graph.get_tensor_by_name(
                        'worker_1/ComputeGradientsG/gradients/worker_1/tinynet/G/conv/conv_/conv/ExpandDims_1_grad/Reshape:0'),
                    graph.get_tensor_by_name(
                        'worker_2/ComputeGradientsG/gradients/worker_2/tinynet/G/conv/conv_/conv/ExpandDims_1_grad/Reshape:0'),
                    graph.get_tensor_by_name(
                        'worker_3/ComputeGradientsG/gradients/worker_3/tinynet/G/conv/conv_/conv/ExpandDims_1_grad/Reshape:0'),
                    graph.get_tensor_by_name(
                        'ApplyGradients/AveOverDevices_14:0')
                ])
                self.check_gradients(gen_0, gen_1, gen_2, gen_3, gen_ave)
                self.check_gradients(dis_0, dis_1, dis_2, dis_3, dis_ave)
            SESS_FINISHED.send(test_driver.app, itermsg=None)
            test_driver.app.stop()

    def check_gradients(self, g_0, g_1, g_2, g_3, g_ave):
        msg = 'same gradients for different devices'
        self.assertGreater(np.sum(np.abs(g_0 - g_1)), 0.0, msg)
        self.assertGreater(np.sum(np.abs(g_0 - g_2)), 0.0, msg)
        self.assertGreater(np.sum(np.abs(g_0 - g_3)), 0.0, msg)
        self.assertGreater(np.sum(np.abs(g_1 - g_2)), 0.0, msg)
        self.assertGreater(np.sum(np.abs(g_1 - g_3)), 0.0, msg)
        self.assertGreater(np.sum(np.abs(g_2 - g_3)), 0.0, msg)
        g_array = np.concatenate([g_0.reshape((1, -1)),
                                  g_1.reshape((1, -1)),
                                  g_2.reshape((1, -1)),
                                  g_3.reshape((1, -1))], axis=0)
        g_ave = g_ave.reshape(-1)
        g_np_ave = np.mean(g_array, axis=0)
        self.assertAllClose(g_np_ave, g_ave)

    def test_rand_initialisation(self):
        test_driver = get_initialised_driver(0, True)
        graph = test_driver.create_graph(test_driver.app, 1, True)
        with self.cached_session(graph=graph) as sess:
            test_tensor = graph.get_tensor_by_name(
                "G/conv_bn_selu/conv_/w:0")
            with self.assertRaisesRegexp(
                    tf.errors.FailedPreconditionError,
                    'uninitialized value'):
                sess.run(test_tensor)
            ModelRestorer(**vars(test_driver)).rand_init_model(None)
            sess.run(test_tensor)
            _ = sess.run(tf.global_variables())

    def test_from_latest_file_initialisation(self):
        test_driver = get_initialised_driver(-1, False)
        expected_init = np.array(
            [[-0.03544217, 0.0228963, -0.04585603, 0.16923568, -0.51635778,
              0.60694504, 0.01968583, -0.6252712, 0.28622296, -0.29527491,
              0.61191976, 0.27878678, -0.07661559, -0.41357407, 0.70488983,
              -0.10836645, 0.06488426, 0.0746650, -0.188567, -0.64652514]],
            dtype=np.float32)
        graph = test_driver.create_graph(test_driver.app, 1, True)
        with self.cached_session(graph=graph) as sess:
            test_tensor = graph.get_tensor_by_name(
                "G/conv_bn_selu/conv_/w:0")
            with self.assertRaisesRegexp(
                    tf.errors.FailedPreconditionError,
                    'uninitialized value'):
                _ = sess.run(test_tensor)
            ModelRestorer(**vars(test_driver)).restore_model(None)
            after_init = sess.run(test_tensor)
            self.assertAllClose(after_init[0], expected_init)
            _ = sess.run(tf.global_variables())

    # def test_not_found_file_initialisation(self):
    #     test_driver = get_initialised_driver(42, False)
    #     graph = test_driver.create_graph(test_driver.app, 1, True)
    #     with self.cached_session(graph=graph) as sess:
    #         with self.assertRaisesRegexp(
    #                 ValueError, ''):
    #             ModelRestorer(**vars(test_driver)).restore_model(None)
    #         with self.assertRaisesRegexp(
    #                 tf.errors.NotFoundError, 'Failed to find'):
    #             ModelRestorer(**vars(test_driver)).restore_model(None)

    def test_from_file_initialisation(self):
        test_driver = get_initialised_driver(40, False)
        expected_init = np.array(
            [[-0.23192197, 0.60880029, -0.24921742, -0.00186354, -0.3345384,
              0.16067748, -0.2210995, -0.19460233, -0.3035436, -0.42839912,
              -0.0489039, -0.90753943, -0.12664583, -0.23129687, 0.01584663,
              -0.43854219, 0.40412974, 0.0396539, -0.1590578, -0.53759819]],
            dtype=np.float32)
        graph = test_driver.create_graph(test_driver.app, 1, True)
        with self.cached_session(graph=graph) as sess:
            test_tensor = graph.get_tensor_by_name(
                "G/conv_bn_selu/conv_/w:0")
            with self.assertRaisesRegexp(
                    tf.errors.FailedPreconditionError,
                    'uninitialized value'):
                _ = sess.run(test_tensor)
            ModelRestorer(**vars(test_driver)).restore_model(None)
            after_init = sess.run(test_tensor)
            self.assertAllClose(after_init[0], expected_init)
            _ = sess.run(tf.global_variables())

    def test_from_file_finetuning(self):
        test_driver = get_initialised_driver(
            40, False, '.*conv_bn_selu/.*conv_.*')
        expected_init = np.array(
            [[-0.23192197, 0.60880029, -0.24921742, -0.00186354, -0.3345384,
              0.16067748, -0.2210995, -0.19460233, -0.3035436, -0.42839912,
              -0.0489039, -0.90753943, -0.12664583, -0.23129687, 0.01584663,
              -0.43854219, 0.40412974, 0.0396539, -0.1590578, -0.53759819]],
            dtype=np.float32)
        graph = test_driver.create_graph(test_driver.app, 1, True)
        with self.cached_session(graph=graph) as sess:
            test_tensor = graph.get_tensor_by_name(
                "G/conv_bn_selu/conv_/w:0")
            test_negative_tensor = graph.get_tensor_by_name(
                "D/conv_relu/conv_/b:0")
            with self.assertRaisesRegexp(
                    tf.errors.FailedPreconditionError,
                    'uninitialized value'):
                _ = sess.run(test_tensor)
            ModelRestorer(**vars(test_driver)).restore_model(None)
            after_init = sess.run(test_tensor)
            after_init_negative = sess.run(test_negative_tensor)
            self.assertAllClose(after_init[0], expected_init)
            # the not matched should be initialised using default initializer
            self.assertEqual(np.any(after_init_negative), False)
            _ = sess.run(tf.global_variables())
            bad_init = sess.run(tf.report_uninitialized_variables())
            self.assertEqual(bad_init.size, 0)


if __name__ == "__main__":
    set_logger()
    # _run_test_application()
    tf.test.main()
