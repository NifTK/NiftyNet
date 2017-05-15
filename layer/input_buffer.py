# -*- coding: utf-8 -*-
"""
This module define queues that stores training/evaluation images (and labels)
TrainEvalInputBuffer provides randomised queue

DeployInputBuffer provides FIFO queue. This is designed for making
patch-based predictions for mulitple test volumes.
"""

import threading
import time

import tensorflow as tf

from .input_sampler import ImageSampler


class InputBatchQueueRunner(object):
    """
    This class defines a light wrapper around queue objects
    for training pair/evaluation pairs.
    After initialisation, run_threads() can be called with tf.session and
    tf.coordinator to start generating samples with multiple threads.

    The sampling threads can be stopped by:
        1. close_all() called externally -- all threads quit immediately
        2. no more elements from self.sampler() -- the last
           thread will call close_all() with a grace period 60 seconds
    """

    def __init__(self, batch_size, capacity, sampler, shuffle=True):

        assert isinstance(sampler, ImageSampler)
        assert batch_size > 0
        self.batch_size = batch_size
        self.sampler = sampler

        # define queue properties
        self.capacity = max(capacity, batch_size * 2)
        self.shuffle = shuffle
        self.min_queue_size = self.capacity / 2 if self.shuffle else 0

        # create queue and the associated operations
        self._queue = None
        self._enqueue_op = None
        self._dequeue_op = None
        self._query_queue_size_op = None
        self._close_queue_op = None
        self._create_queue_and_ops()

        # keep track of session and threads created by this class instance
        self._session = None
        self._coordinator = None
        self._threads = []

        # this variable is used to monitor the sampler threads
        self._started_threads = []

    def _create_queue_and_ops(self):
        """
        Create a shuffled queue or FIFOqueue, and create queue
        operations. This should be called before tf.Graph.finalize.
        """

        # create queue
        if self.shuffle:
            self._queue = tf.RandomShuffleQueue(
                capacity=self.capacity,
                min_after_dequeue=self.min_queue_size,
                dtypes=self.sampler.placeholder_dtypes,
                shapes=self.sampler.placeholder_shapes,
                names=self.sampler.placeholder_names,
                name="shuffled_queue")
        else:
            self._queue = tf.FIFOQueue(  # pylint: disable=redefined-variable-type
                capacity=self.capacity,
                dtypes=self.sampler.placeholder_dtypes,
                shapes=self.sampler.placeholder_shapes,
                names=self.sampler.placeholder_names,
                name="FIFO_queue")

        # create queue operations
        queue_input_dict = dict(zip(self.sampler.placeholder_names,
                                    self.sampler.placeholders))
        self._enqueue_op = self._queue.enqueue(queue_input_dict)
        self._dequeue_op = self._queue.dequeue_many(self.batch_size)
        self._query_queue_size_op = self._queue.size()
        self._close_queue_op = self._queue.close(cancel_pending_enqueues=True)

    def _push(self, thread_id):
        print('New thread: {}'.format(thread_id))
        try:
            if self.shuffle:
                iterator = self.sampler()
            else:
                iterator = self.sampler(self.batch_size)

            for t in iterator:
                if self._session._closed:
                    break
                if self._coordinator.should_stop():
                    break
                # ensure the shape of sampler's output matches
                # the queue input definitions
                assert len(t.keys()[0]) == len(t.values()[0])
                self._session.run(self._enqueue_op, feed_dict=t)
        except tf.errors.CancelledError:
            pass
        except AssertionError as e:
            print "AssertionError: please check the sampler() dims"
        except ValueError as e:
            print e
        except RuntimeError as e:
            print e
        finally:
            # this thread won't enqueue anymore
            self._started_threads[thread_id] = False
            # try to close down when it's the last thread
            if not any(self._started_threads):
                # preparing closing down
                # waiting to be sure the last few batches are dequeued
                retry, interval = 60000, 0.001
                print "stopping the sampling threads... " \
                      "(in {} seconds)".format(retry * interval)
                remained = self.current_queue_size - self.min_queue_size
                while retry > 0:
                    if self.batch_size > remained:
                        break
                    # more batches can be processed before deleting the queue
                    time.sleep(interval)
                    retry -= 1
                    remained = self.current_queue_size - self.min_queue_size
                if remained > 0:
                    # still having elements left
                    print("Insufficient samples {} to form a {}-element " \
                          "batch in queue".format(self.batch_size, remained))
                # close the queue
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
        self._started_threads = [False for i in range(num_threads)]
        for idx in range(num_threads):
            self._threads.append(
                threading.Thread(target=self._push, args=[idx]))
            self._threads[idx].daemon = True
            self._threads[idx].start()
            self._started_threads[idx] = True

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
                                   stop_grace_period_secs=0,
                                   ignore_live_threads=True)
        except RuntimeError as e:
            print e
        finally:
            if not self._session._closed:
                self._session.run(self._close_queue_op)


class DeployInputBuffer(InputBatchQueueRunner):
    def __init__(self, batch_size, capacity, sampler):
        super(DeployInputBuffer, self).__init__(batch_size=batch_size,
                                                capacity=capacity,
                                                sampler=sampler,
                                                shuffle=False)


class TrainEvalInputBuffer(InputBatchQueueRunner):
    def __init__(self, batch_size, capacity, sampler, shuffle=True):
        super(TrainEvalInputBuffer, self).__init__(batch_size=batch_size,
                                                   capacity=capacity,
                                                   sampler=sampler,
                                                   shuffle=shuffle)
