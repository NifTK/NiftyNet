# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.engine.application_variables import GradientsCollector
from niftynet.engine.handler_gradient import ApplyGradients
from niftynet.network.toynet import ToyNet
from tests.niftynet_testcase import NiftyNetTestCase

def get_test_network():
    net = ToyNet(num_classes=4)
    return net


class GradientCollectorTest(NiftyNetTestCase):
    def test_nested_gradients(self):
        n_device = 3
        grad_collector = GradientsCollector(n_devices=n_device)

        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx) as scope:
                image = tf.ones([2, 32, 32, 32, 4], dtype=tf.float32)
                test_net = get_test_network()(image, is_training=True)
                loss = tf.reduce_mean(tf.square(test_net - image))
                optimiser = tf.train.GradientDescentOptimizer(0.1)
                grads = optimiser.compute_gradients(loss)
                grad_collector.add_to_collection([grads])
        self.assertAllClose(len(grad_collector._gradients), n_device)
        with self.assertRaisesRegexp(AssertionError, ""):
            grad_collector.add_to_collection(grads)
        ave_grads = grad_collector.gradients
        self.assertAllClose(len(grad_collector._gradients[0]), len(ave_grads))
        self.assertAllClose(
            grad_collector._gradients[0][0][0][0].shape.as_list(),
            ave_grads[0][0][0].shape.as_list())

    def test_gradients(self):
        n_device = 3
        grad_collector = GradientsCollector(n_devices=n_device)

        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx) as scope:
                image = tf.ones([2, 32, 32, 32, 4], dtype=tf.float32)
                test_net = get_test_network()(image, is_training=True)
                loss = tf.reduce_mean(tf.square(test_net - image))
                optimiser = tf.train.GradientDescentOptimizer(0.1)
                grads = optimiser.compute_gradients(loss)
                grad_collector.add_to_collection(grads)
        self.assertAllClose(len(grad_collector._gradients), n_device)
        with self.assertRaisesRegexp(AssertionError, ""):
            grad_collector.add_to_collection(grads)
        ave_grads = grad_collector.gradients
        self.assertAllClose(len(grad_collector._gradients[0]), len(ave_grads))
        self.assertAllClose(
            grad_collector._gradients[0][0][0][0].shape.as_list(),
            ave_grads[0][0][0].shape.as_list())

    def test_multiple_loss_gradients(self):
        n_device = 3
        grad_collector = GradientsCollector(n_devices=n_device)

        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx) as scope:
                image = tf.ones([2, 32, 32, 32, 4], dtype=tf.float32)
                test_net = get_test_network()(image, is_training=True)
                loss = tf.reduce_mean(tf.square(test_net - image))
                loss_1 = tf.reduce_mean(tf.abs(test_net - image))
                optimiser = tf.train.GradientDescentOptimizer(0.1)
                grads = optimiser.compute_gradients(loss)
                grads_1 = optimiser.compute_gradients(loss_1)
                grad_collector.add_to_collection([grads, grads_1])
        self.assertAllClose(len(grad_collector._gradients), n_device)
        with self.assertRaisesRegexp(AssertionError, ""):
            grad_collector.add_to_collection(grads)
        ave_grads = grad_collector.gradients
        self.assertAllClose(len(grad_collector._gradients[0]), len(ave_grads))
        self.assertAllClose(
            grad_collector._gradients[0][0][0][0].shape.as_list(),
            ave_grads[0][0][0].shape.as_list())

    def test_multiple_device_multiple_loss_gradients_with_multiple_optimiser(self):
        n_device = 3
        grad_collector = GradientsCollector(n_devices=n_device)

        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx) as scope:
                image = tf.ones([2, 32, 32, 32, 4], dtype=tf.float32)
                test_net = get_test_network()(image, is_training=True)
                loss = tf.reduce_mean(tf.square(test_net - image))
                loss_1 = tf.reduce_mean(tf.abs(test_net - image))
                grads = dict()
                optimiser = dict()
                optimiser['opt'] = tf.train.GradientDescentOptimizer(0.1)
                optimiser['opt_1'] = tf.train.GradientDescentOptimizer(0.1)
                grads['opt'] = optimiser['opt'].compute_gradients(loss)
                grads['opt_1'] = optimiser['opt_1'].compute_gradients(loss_1)
                grad_collector.add_to_collection(grads)
        self.assertAllClose(len(grad_collector._gradients), n_device)
        with self.assertRaisesRegexp(AssertionError, ""):
            grad_collector.add_to_collection(grads)
        ave_grads = grad_collector.gradients
        self.assertAllClose(len(grad_collector._gradients[0]), len(ave_grads))
        self.assertAllClose(
            grad_collector._gradients[0]['opt'][0][0][0].shape.as_list(),
            ave_grads['opt'][0][0][0].shape.as_list())
        self.assertAllClose(
            grad_collector._gradients[0]['opt_1'][0][0][0].shape.as_list(),
            ave_grads['opt_1'][0][0][0].shape.as_list())

    def test_single_device_gradients(self):
        n_device = 1
        grad_collector = GradientsCollector(n_devices=n_device)

        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx) as scope:
                image = tf.ones([2, 32, 32, 32, 4], dtype=tf.float32)
                test_net = get_test_network()(image, is_training=True)
                loss = tf.reduce_mean(tf.square(test_net - image))
                loss_1 = tf.reduce_mean(tf.abs(test_net - image))
                optimiser = tf.train.GradientDescentOptimizer(0.1)
                grads = optimiser.compute_gradients(loss)
                grads_1 = optimiser.compute_gradients(loss_1)
                grad_collector.add_to_collection([grads, grads_1])
        self.assertAllClose(len(grad_collector._gradients), n_device)
        with self.assertRaisesRegexp(AssertionError, ""):
            grad_collector.add_to_collection(grads)
        ave_grads = grad_collector.gradients
        self.assertAllClose(len(grad_collector._gradients[0]), len(ave_grads))
        self.assertAllClose(
            grad_collector._gradients[0][0][0][0].shape.as_list(),
            ave_grads[0][0][0].shape.as_list())

    def test_simple_gradients(self):
        n_device = 1
        grad_collector = GradientsCollector(n_devices=n_device)

        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx) as scope:
                image = tf.ones([2, 32, 32, 32, 4], dtype=tf.float32)
                test_net = get_test_network()(image, is_training=True)
                loss = tf.reduce_mean(tf.square(test_net - image))
                optimiser = tf.train.GradientDescentOptimizer(0.1)
                grads = optimiser.compute_gradients(loss)
                grad_collector.add_to_collection(grads)
        self.assertAllClose(len(grad_collector._gradients), n_device)
        with self.assertRaisesRegexp(AssertionError, ""):
            grad_collector.add_to_collection(grads)
        ave_grads = grad_collector.gradients
        self.assertAllClose(len(grad_collector._gradients[0]), len(ave_grads))
        self.assertAllClose(
            grad_collector._gradients[0][0][0].shape.as_list(),
            ave_grads[0][0].shape.as_list())


if __name__ == "__main__":
    tf.test.main()
