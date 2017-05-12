# -*- coding: utf-8 -*-
"""
This module define queues that stores training/evaluation images (and labels)
TrainEvalInputBuffer provides randomised queue

DeployInputBuffer provides FIFO queue. This is designed for making
patch-based predictions for mulitple test volumes.
"""
import time
import threading
import tensorflow as tf


class InputBatchQueueRunner(object):
    """
    This class defines a light wrapper around queue objects
    for training pair/evaluation pairs.
    After initialisation, run_threads() can be called with tf.session and
    tf.coordinator to start generating samples with multiple threads.

    The sampling threads can be stopped by:
        1. close_all() called externally -- all threads quit immediately
        2. no more elements from self.element_generator() -- the last
           thread will call close_all() with a grace period 60 seconds
    """

    def __init__(self,
                 batch_size,
                 capacity,
                 input_names,
                 input_types,
                 input_shapes,
                 generator,
                 shuffle=True):

        assert callable(generator)
        assert batch_size <= (capacity/2)
        assert batch_size > 0
        assert len(input_names) == len(input_types)
        assert len(input_names) == len(input_shapes)
        self.batch_size = batch_size
        self.element_generator = generator

        # define queue properties
        self.input_names, self.input_types, self.input_shapes = \
                input_names, input_types, input_shapes
        self.capacity = capacity
        self.shuffle = shuffle
        self.min_queue_size = self.capacity / 2 if self.shuffle else 0

        # create queue and associated operations
        self._queue = None
        self._place_holders = None
        self._enqueue_op = None
        self._dequeue_op = None
        self._query_queue_size_op = None
        self._close_queue_op = None
        self._create_queue_and_ops()

        # keep track of session and threads created by this class instance
        self._session = None
        self._coordinator = None
        self._threads = []
        # this variable is used to monitor the generator threads
        self._started_threads = []

    def _create_queue_and_ops(self):
        """
        Create a shuffled queue or FIFOqueue, and create queue
        operations. These should be called before tf.Graph.finalize.
        """
        # create queue
        if self.shuffle:
            self._queue = tf.RandomShuffleQueue(
                capacity=self.capacity,
                min_after_dequeue=self.min_queue_size,
                dtypes=self.input_types,
                shapes=self.input_shapes,
                names=self.input_names,
                name="shuffled_queue")
        else:
            self._queue = tf.FIFOQueue(# pylint: disable=redefined-variable-type
                capacity=self.capacity,
                dtypes=self.input_types,
                shapes=self.input_shapes,
                names=self.input_names,
                name="FIFO_queue")
        # create place holders
        self._place_holders = tuple(tf.placeholder(dtype,
                                                   shape=self.input_shapes[i],
                                                   name=self.input_names[i])
                                    for i, dtype in enumerate(self.input_types))
        # create enqueue operation
        queue_input_dict = dict(zip(self.input_names, self._place_holders))
        self._enqueue_op = self._queue.enqueue(queue_input_dict)
        self._dequeue_op = self._queue.dequeue_many(self.batch_size)
        self._query_queue_size_op = self._queue.size()
        self._close_queue_op = self._queue.close(cancel_pending_enqueues=True)

    def _push(self, thread_id):
        try:
            for t in self.element_generator():
                if self._session._closed:
                    break
                if self._coordinator.should_stop():
                    break
                self._session.run(self._enqueue_op,
                                  feed_dict={self._place_holders: t})
            # this thread won't enqueue anymore
            self._started_threads[thread_id] = False

        except tf.errors.CancelledError:
            pass
        except ValueError as e:
            print e
        except RuntimeError as e:
            print e
        finally:
            # try to close down when it's the last thread
            if not any(self._started_threads):
                # preparing closing down
                # waiting to be sure the last few batches are dequeued
                retry, interval = 60000, 0.001
                print "stopping the sampling threads..."\
                      "({} seconds grace period)".format(retry * interval)
                while retry > 0:
                    remained = self.current_queue_size - self.min_queue_size
                    if self.batch_size > remained:
                        break
                    # more batches can be processed before deleting the queue
                    time.sleep(interval)
                    retry -= 1
                if remained > 0:
                    # still having elements left, we can't do anything with that
                    print("Insufficient samples to form a {}-element batch: "\
                          "{} available in queue".format(
                              self.batch_size, remained))
                self.close_all()

    @property
    def current_queue_size(self):
        if self._session._closed:
            return 0
        return self._session.run(self._query_queue_size_op)

    @property
    def pop_batch_op(self):
        return self._dequeue_op

    def run_threads(self, session, coord, num_threads=1):
        print 'Starting preprocessing threads...'
        self._session = session
        self._coordinator = coord
        for idx in range(num_threads):
            self._threads.append(
                threading.Thread(target=self._push, args=[idx]))
            self._threads[idx].daemon = True
            self._threads[idx].start()
            self._started_threads.append(True)

    def close_all(self):
        if not self._threads:
            raise RuntimeError("the queue is not currently running")
        try:
            self._coordinator.request_stop()
            self._coordinator.join(threads=self._threads,
                                   stop_grace_period_secs=0,
                                   ignore_live_threads=True)
        except RuntimeError as e:
            print e
        finally:
            if not self._session._closed:
                self._session.run(self._close_queue_op)


class DeployInputBuffer(InputBatchQueueRunner):
    def __init__(self,
                 batch_size,
                 capacity,
                 shapes,
                 sample_generator):
        input_names = ("images", "info")
        input_types = (tf.float32, tf.int64)
        super(DeployInputBuffer, self).__init__(batch_size=batch_size,
                                                capacity=capacity,
                                                input_names=input_names,
                                                input_types=input_types,
                                                input_shapes=shapes,
                                                generator=sample_generator,
                                                shuffle=False)


class TrainEvalInputBuffer(InputBatchQueueRunner):
    def __init__(self,
                 batch_size,
                 capacity,
                 shapes,
                 sample_generator,
                 shuffle=True):
        input_names = ("images", "labels", "info")
        input_types = (tf.float32, tf.int64, tf.int64)
        super(TrainEvalInputBuffer, self).__init__(batch_size=batch_size,
                                                   capacity=capacity,
                                                   input_names=input_names,
                                                   input_types=input_types,
                                                   input_shapes=shapes,
                                                   generator=sample_generator,
                                                   shuffle=shuffle)
