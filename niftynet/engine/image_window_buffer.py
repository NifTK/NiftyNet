# -*- coding: utf-8 -*-
"""
This module define queues that stores training/evaluation images (and labels)
"""

from __future__ import absolute_import, print_function, division

import threading

import numpy as np
import tensorflow as tf

from niftynet.io.misc_io import squeeze_spatial_temporal_dim


class InputBatchQueueRunner(object):
    """
    This class defines a light wrapper around queue objects
    for input windows, and the coordinates describes the original location
    of the window

    After initialisation, run_threads() can be called with tf.session and
    tf.coordinator to start generating samples with multiple threads.

    The sampling threads can be stopped by:
    close_all() called externally -- all threads quit immediately
    """

    def __init__(self, capacity, shuffle=True):
        # define queue properties
        self.capacity = capacity

        self.shuffle = shuffle
        self._batch_size = 1

        # create queue and the associated operations
        self._queue = None
        self._enqueue_op = None
        self._dequeue_func = None
        self._query_queue_size_op = None
        self._close_queue_op = None

        # keep track of session and threads created by this class instance
        self._session = None
        self._coordinator = None
        self._threads = []

        self._window = None

    def _create_queue_and_ops(self, window, enqueue_size=1, dequeue_size=1):
        """
        Create a shuffled queue or FIFO queue, and create queue
        operations. This should be called before tf.Graph.finalize.
        """
        self._window = window
        is_dynamic_window = window.has_dynamic_shapes
        if is_dynamic_window and enqueue_size > 1:
            tf.logging.warning("using dynamic window size, buffer input size "
                               "is set to 1")
        if is_dynamic_window and dequeue_size > 1:
            tf.logging.warning("using dynamic window size, network batch size "
                               "is set to 1")
        self._batch_size = 1 if is_dynamic_window else dequeue_size
        _enqueue_size = 1 if is_dynamic_window else enqueue_size
        assert dequeue_size <= self.capacity, \
            "batch size is larger than the buffer size, " \
            "please increase the queue length or decrease the batch size"

        placeholders_dict = window.placeholders_dict(n_samples=_enqueue_size)
        names = list(placeholders_dict)
        placeholders = list(placeholders_dict.values())
        input_dtypes = [holder.dtype for holder in placeholders]
        input_shapes = [holder.shape[1:] for holder in placeholders] \
            if not is_dynamic_window else None

        # create a queue
        # pylint: disable=redefined-variable-type
        if self.shuffle:
            self._queue = tf.RandomShuffleQueue(
                capacity=self.capacity,
                min_after_dequeue=self.capacity // 2 if self.shuffle else 0,
                dtypes=input_dtypes,
                shapes=input_shapes,
                names=names,
                name="shuffled_queue")
        else:
            self._queue = tf.FIFOQueue(
                capacity=self.capacity,
                dtypes=input_dtypes,
                shapes=input_shapes,
                names=names,
                name="FIFO_queue")

        # create queue operations
        if is_dynamic_window:
            self._enqueue_op = self._queue.enqueue(placeholders_dict)
            self._dequeue_func = self._queue.dequeue
        else:
            self._enqueue_op = self._queue.enqueue_many(placeholders_dict)
            self._dequeue_func = self._queue.dequeue_many
        self._query_queue_size_op = self._queue.size()
        self._close_queue_op = self._queue.close(cancel_pending_enqueues=True)

    def _push(self, thread_id):
        tf.logging.info('New thread: %d', thread_id)
        try:
            output_dict = None
            for output_dict in self():  # pylint: disable=not-callable
                if self._session._closed:
                    break
                if self._coordinator.should_stop():
                    break
                self._session.run(self._enqueue_op, feed_dict=output_dict)

            # push a set of stopping patches
            for _ in range(self.capacity + self._batch_size):
                if self._session._closed:
                    break
                if self._coordinator.should_stop():
                    break
                for name in list(output_dict):
                    output_dict[name] = np.ones_like(output_dict[name]) * -1
                self._session.run(self._enqueue_op, feed_dict=output_dict)

        except tf.errors.CancelledError:
            pass
        # except ValueError as e:
        #    tf.logging.fatal(e)
        #    self.close_all()
        # except RuntimeError as e:
        #    tf.logging.fatal(e)
        #    self.close_all()
        except RuntimeError:
            import sys
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout)
            self.close_all()
        finally:
            pass

    def _current_queue_size(self):
        # for debugging purpose
        if self._session._closed:
            return 0
        return self._session.run(self._query_queue_size_op)

    def pop_batch_op(self):
        """
        This function is used when connecting a sampler output
        to a network. e.g.,
            data_dict = self.get_sampler()[0].pop_batch_op(device_id)
            net_output = net_model(data_dict, is_training)
        Note it squeezes the output tensor of 6 dims
        [batch, x, y, z, time, modality]
        by removing all dims along which length is one.

        :param device_id: integer representing the GPU
        :return: a tensorflow graph op
        """
        if self._window.has_dynamic_shapes:
            data_output = self._dequeue_func()
        else:
            data_output = self._dequeue_func(self._batch_size)
        for (name, shape) in self._window.shapes.items():
            data_output[name].set_shape([self._batch_size] + list(shape))

        for name in data_output:
            data_output[name] = \
                squeeze_spatial_temporal_dim(data_output[name])
        return data_output

    def run_threads(self, session, coord, num_threads=1):
        """
        This function should be called by application.driver,
        where a session and coordinator is maintained, it
        starts sampling threads to fill the queue.

        Note that the threads will be blocked if there's no
        dequeue_op runnning, or number of samples is less
        than the dequeue batch size.

        :param session: a tensorflow session
        :param coord: a tensorflow coordinator
        :param num_threads: integer specifies the number of threads
        :return:
        """
        tf.logging.info('Starting preprocessing threads...')
        self._session = session
        self._coordinator = coord
        for idx in range(num_threads):
            self._threads.append(
                threading.Thread(target=self._push, args=[idx]))
            self._threads[idx].daemon = True
            self._threads[idx].start()

    def close_all(self):
        """
        This function stops all threads immediately and close the queue.
        Further enqueue/dequeue operation raises errors
        """

        if not self._threads:
            raise RuntimeError("the queue is not currently running")
        try:
            self._coordinator.request_stop()
            self._coordinator.join(threads=self._threads,
                                   stop_grace_period_secs=0)
        except RuntimeError:
            pass
        finally:
            if not self._session._closed:
                self._session.run(self._close_queue_op)
