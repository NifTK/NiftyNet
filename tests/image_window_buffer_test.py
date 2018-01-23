# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import ImageWindow
from niftynet.engine.image_window_buffer import InputBatchQueueRunner
from niftynet.utilities.util_common import ParserNamespace


def get_static_image_window():
    window = ImageWindow.from_data_reader_properties(
        source_names={
            'image': (u'modality1', u'modality2'),
            'label': (u'modality3',)},
        image_shapes={
            'image': (192, 160, 192, 1, 2),
            'label': (192, 160, 192, 1, 1)},
        image_dtypes={
            'image': tf.float32,
            'label': tf.float32},
        data_param={
            'modality1': ParserNamespace(spatial_window_size=(10, 10, 2)),
            'modality2': ParserNamespace(spatial_window_size=(10, 10, 2)),
            'modality3': ParserNamespace(spatial_window_size=(5, 5, 1))}
    )
    return window


def get_dynamic_image_window():
    window = ImageWindow.from_data_reader_properties(
        source_names={
            'image': (u'modality1', u'modality2'),
            'label': (u'modality3',)},
        image_shapes={
            'image': (192, 160, 192, 1, 2),
            'label': (192, 160, 192, 1, 1)},
        image_dtypes={
            'image': tf.float32,
            'label': tf.float32},
        data_param={
            'modality1': ParserNamespace(spatial_window_size=(10, 10)),
            'modality2': ParserNamespace(spatial_window_size=(10, 10)),
            'modality3': ParserNamespace(spatial_window_size=(5, 5, 1))}
    )
    return window


class ImageWindowBufferTest(tf.test.TestCase):
    def test_static_window_init(self):
        window = get_static_image_window()
        self.assertAllEqual(window.has_dynamic_shapes, False)

        test_buffer = InputBatchQueueRunner(capacity=10, shuffle=True)
        with self.assertRaisesRegexp(AttributeError, ""):
            test_buffer._create_queue_and_ops('test')
        test_buffer._create_queue_and_ops(window)
        self.assertIsInstance(test_buffer._queue, tf.RandomShuffleQueue)
        self.assertAllEqual(test_buffer._enqueue_op.name,
                            'shuffled_queue_EnqueueMany')
        self.assertAllEqual(test_buffer._enqueue_op.name,
                            'shuffled_queue_EnqueueMany')

        test_buffer = InputBatchQueueRunner(capacity=10, shuffle=False)
        test_buffer._create_queue_and_ops(window)
        self.assertIsInstance(test_buffer._queue, tf.FIFOQueue)
        self.assertAllEqual(test_buffer._enqueue_op.name,
                            'FIFO_queue_EnqueueMany')
        self.assertAllEqual(test_buffer._enqueue_op.name,
                            'FIFO_queue_EnqueueMany')

        with self.assertRaisesRegexp(NotImplementedError, ""):
            test_buffer()

    def test_dynamic_window_init(self):
        window = get_dynamic_image_window()
        self.assertAllEqual(window.has_dynamic_shapes, True)

        test_buffer = InputBatchQueueRunner(capacity=10, shuffle=True)
        with self.assertRaisesRegexp(AttributeError, ""):
            test_buffer._create_queue_and_ops('test')
        test_buffer._create_queue_and_ops(window)
        self.assertIsInstance(test_buffer._queue, tf.RandomShuffleQueue)
        self.assertAllEqual(test_buffer._enqueue_op.name,
                            'shuffled_queue_enqueue')
        self.assertAllEqual(test_buffer._enqueue_op.name,
                            'shuffled_queue_enqueue')

        test_buffer = InputBatchQueueRunner(capacity=10, shuffle=False)
        test_buffer._create_queue_and_ops(window)
        self.assertIsInstance(test_buffer._queue, tf.FIFOQueue)
        self.assertAllEqual(test_buffer._enqueue_op.name,
                            'FIFO_queue_enqueue')
        self.assertAllEqual(test_buffer._enqueue_op.name,
                            'FIFO_queue_enqueue')

        with self.assertRaisesRegexp(NotImplementedError, ""):
            test_buffer()

    def test_static_window_enqueue(self):
        enqueue_size = 3
        dequeue_size = 2
        test_buffer = InputBatchQueueRunner(capacity=10, shuffle=True)

        window = get_static_image_window()
        test_buffer._create_queue_and_ops(window,
                                          enqueue_size=enqueue_size,
                                          dequeue_size=dequeue_size)

        enqueue_dict = {}
        placeholder = window.image_data_placeholder('image')
        enqueue_dict[placeholder] = np.zeros(placeholder.shape.as_list())
        placeholder = window.coordinates_placeholder('image')
        enqueue_dict[placeholder] = np.zeros(placeholder.shape.as_list())
        placeholder = window.image_data_placeholder('label')
        enqueue_dict[placeholder] = np.zeros(placeholder.shape.as_list())
        placeholder = window.coordinates_placeholder('label')
        enqueue_dict[placeholder] = np.zeros(placeholder.shape.as_list())

        with self.test_session() as sess:
            # queue size before enqueue
            queue_size = sess.run(test_buffer._query_queue_size_op)
            self.assertAllEqual(0, queue_size)
            # do enqueue
            sess.run(test_buffer._enqueue_op, feed_dict=enqueue_dict)
            # queue size after enqueue
            queue_size = sess.run(test_buffer._query_queue_size_op)
            self.assertAllEqual(enqueue_size, queue_size)

            for _ in range(2):
                sess.run(test_buffer._enqueue_op, feed_dict=enqueue_dict)
            queue_size = sess.run(test_buffer._query_queue_size_op)
            self.assertAllEqual(9, queue_size)
            sess.run(test_buffer.pop_batch_op())
            queue_size = sess.run(test_buffer._query_queue_size_op)
            self.assertAllEqual(7, queue_size)
            sess.run(test_buffer._close_queue_op)

    def test_dynamic_window_enqueue(self):
        enqueue_size = 3
        dequeue_size = 2
        test_buffer = InputBatchQueueRunner(capacity=10, shuffle=False)

        window = get_dynamic_image_window()
        test_buffer._create_queue_and_ops(window, enqueue_size, dequeue_size)

        dynamic_image_size = {'image': (5, 5, 5, 1, 2),
                              'label': (5, 5, 3, 1, 2)}
        window_shape = window.match_image_shapes(dynamic_image_size)

        enqueue_dict = {}
        placeholder = window.image_data_placeholder('image')
        enqueue_dict[placeholder] = np.zeros((1,) + window_shape['image'])
        placeholder = window.coordinates_placeholder('image')
        enqueue_dict[placeholder] = np.zeros(placeholder.shape.as_list())
        placeholder = window.image_data_placeholder('label')
        enqueue_dict[placeholder] = np.zeros((1,) + window_shape['label'])
        placeholder = window.coordinates_placeholder('label')
        enqueue_dict[placeholder] = np.zeros(placeholder.shape.as_list())

        with self.test_session() as sess:
            # queue size before enqueue
            expected_queue_size = 0
            queue_size = sess.run(test_buffer._query_queue_size_op)
            self.assertAllEqual(expected_queue_size, queue_size)
            # do enqueue
            sess.run(test_buffer._enqueue_op, feed_dict=enqueue_dict)
            # queue size after enqueue
            expected_queue_size = 1
            queue_size = sess.run(test_buffer._query_queue_size_op)
            self.assertAllEqual(expected_queue_size, queue_size)
            # do dequeue
            sess.run(test_buffer.pop_batch_op())
            # queue size after enqueue
            queue_size = sess.run(test_buffer._query_queue_size_op)
            self.assertAllEqual(0, queue_size)
            sess.run(test_buffer._close_queue_op)


if __name__ == "__main__":
    tf.test.main()
