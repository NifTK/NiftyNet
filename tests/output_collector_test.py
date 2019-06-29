# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.engine.application_variables import \
    NETWORK_OUTPUT, CONSOLE, TF_SUMMARIES
from niftynet.engine.application_variables import OutputsCollector
from tests.niftynet_testcase import NiftyNetTestCase

# def get_test_network():
#    net = ToyNet(num_classes=4)
#    return net


class OutputCollectorTest(NiftyNetTestCase):
    def test_add_to_single_device(self):
        n_device = 1
        collector = OutputsCollector(n_devices=n_device)
        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx):
                image = tf.ones([2, 32, 32, 32, 1])
                foo = tf.zeros([2, 2])
                bar = tf.zeros([42])
                collector.add_to_collection(name='image',
                                            var=image,
                                            average_over_devices=False)
                collector.add_to_collection(name='foo',
                                            var=foo,
                                            average_over_devices=False)
                collector.add_to_collection(name='bar',
                                            var=bar,
                                            collection=NETWORK_OUTPUT,
                                            average_over_devices=False)
        self.assertDictEqual(collector.variables(collection=CONSOLE),
                             {'image': image, 'foo': foo})
        self.assertDictEqual(collector.variables(collection=NETWORK_OUTPUT),
                             {'bar': bar})

    def test_add_to_multiple_device(self):
        n_device = 4
        collector = OutputsCollector(n_devices=n_device)
        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx):
                image = tf.ones([2, 32, 32, 32, 1])
                foo = tf.zeros([2, 2])
                bar = tf.zeros([42])
                collector.add_to_collection(name='image',
                                            var=image,
                                            average_over_devices=False)
                collector.add_to_collection(name='foo',
                                            var=foo,
                                            average_over_devices=False)
                collector.add_to_collection(name='bar',
                                            var=bar,
                                            average_over_devices=True)
        self.assertEqual(
            set(collector.variables()),
            {'image_1', 'image_3', 'image_2',
             'image', 'foo_1', 'foo_2', 'foo_3', 'foo', 'bar'})
        self.assertEqual(len(collector.variables()['bar']), n_device)
        collector.finalise_output_op()
        self.assertIsInstance(collector.variables()['bar'], tf.Tensor)

    def test_netout_single_device(self):
        n_device = 1
        collector = OutputsCollector(n_devices=n_device)
        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx):
                image = tf.ones([2, 32, 32, 32, 1])
                foo = tf.zeros([2, 2])
                collector.add_to_collection(name='image',
                                            var=image,
                                            collection=NETWORK_OUTPUT,
                                            average_over_devices=False)
                collector.add_to_collection(name='foo',
                                            var=foo,
                                            collection=NETWORK_OUTPUT,
                                            average_over_devices=False)
            self.assertDictEqual(collector.variables(collection=NETWORK_OUTPUT),
                                 {'image': image, 'foo': foo})

    def test_netout_multiple_device(self):
        n_device = 4
        collector = OutputsCollector(n_devices=n_device)
        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx):
                image = tf.ones([2, 32, 32, 32, 1])
                foo = tf.zeros([2, 2])
                bar = tf.zeros([42])
                collector.add_to_collection(name='image',
                                            var=image,
                                            collection=NETWORK_OUTPUT,
                                            average_over_devices=False)
                collector.add_to_collection(name='foo',
                                            var=foo,
                                            collection=NETWORK_OUTPUT,
                                            average_over_devices=False)
                collector.add_to_collection(name='bar',
                                            var=bar,
                                            collection=NETWORK_OUTPUT,
                                            average_over_devices=True)
        self.assertEqual(
            set(collector.variables(NETWORK_OUTPUT)),
            {'image_1', 'image_3', 'image_2',
             'image', 'foo_1', 'foo_2', 'foo_3', 'foo', 'bar'})
        self.assertEqual(len(collector.variables(NETWORK_OUTPUT)['bar']),
                         n_device)
        collector.finalise_output_op()
        self.assertIsInstance(collector.variables(NETWORK_OUTPUT)['bar'],
                              tf.Tensor)

    def test_tf_summary_single_device(self):
        n_device = 1
        collector = OutputsCollector(n_devices=n_device)
        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx):
                image = tf.ones([2, 32, 32, 32, 1])
                foo = tf.zeros([2, 2])
                collector.add_to_collection(name='image',
                                            var=image,
                                            collection=TF_SUMMARIES,
                                            average_over_devices=False)
                collector.add_to_collection(name='foo',
                                            var=foo,
                                            collection=TF_SUMMARIES,
                                            average_over_devices=False)
        self.assertDictEqual(collector.summary_vars,
                             {'image': image, 'foo': foo})

    def test_tf_summary_multiple_device(self):
        n_device = 4
        collector = OutputsCollector(n_devices=n_device)
        for idx in range(n_device):
            with tf.name_scope('worker_%d' % idx):
                image = tf.ones([2, 32, 32, 32, 1])
                foo = tf.zeros([2, 2])
                bar = tf.zeros([42])
                collector.add_to_collection(name='image',
                                            var=image,
                                            collection=TF_SUMMARIES,
                                            average_over_devices=False)
                collector.add_to_collection(name='foo',
                                            var=foo,
                                            collection=TF_SUMMARIES,
                                            average_over_devices=False)
                collector.add_to_collection(name='bar',
                                            var=bar,
                                            collection=TF_SUMMARIES,
                                            average_over_devices=True)
        self.assertEqual(
            set(collector.summary_vars),
            {'image_1', 'image_3', 'image_2',
             'image', 'foo_1', 'foo_2', 'foo_3', 'foo', 'bar'})
        self.assertEqual(len(collector.summary_vars['bar']), n_device)
        collector.finalise_output_op()
        self.assertIsInstance(collector.summary_vars['bar'], tf.Tensor)

    def test_ill_add(self):
        collector = OutputsCollector(n_devices=2)
        foo = tf.zeros([2, 2])
        bar = tf.zeros([42])
        with self.assertRaisesRegexp(AssertionError, ""):
            collector.add_to_collection(name=None, var=None)
        with self.assertRaisesRegexp(AssertionError, ""):
            collector.add_to_collection(name=None, var=bar)
        with self.assertRaisesRegexp(ValueError, ""):
            collector.add_to_collection(name=foo, var=bar,
                                        average_over_devices=True)
            collector.add_to_collection(name=foo, var=bar,
                                        average_over_devices=True)
            collector.add_to_collection(name=foo, var=bar,
                                        average_over_devices=True)


if __name__ == "__main__":
    tf.test.main()
