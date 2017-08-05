# -*- coding: utf-8 -*-
"""
This module define queues that stores training/evaluation images (and labels)
TrainEvalInputBuffer provides randomised queue

DeployInputBuffer provides FIFO queue. This is designed for making
patch-based predictions for multiple test volumes.
"""

from __future__ import absolute_import
from __future__ import print_function

import threading

import tensorflow as tf

from niftynet.engine.base_sampler import BaseSampler
from niftynet.utilities.input_placeholders import ImagePatch


class InputBatchQueueRunner(object):
    """
    This class defines a light wrapper around queue objects
    for training pair/evaluation pairs.
    After initialisation, run_threads() can be called with tf.session and
    tf.coordinator to start generating samples with multiple threads.

    The sampling threads can be stopped by:
    close_all() called externally -- all threads quit immediately
    """

    def __init__(self, batch_size, capacity, placeholders_dict, shuffle=True):

        assert batch_size > 0
        self.batch_size = batch_size
        # define queue properties
        self.capacity = max(capacity, batch_size * 2)
        self.shuffle = shuffle
        self.min_queue_size = self.capacity // 2 if self.shuffle else 0
        self.__fields = list(placeholders_dict)
        self.__placeholders = list(placeholders_dict.values())

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


    def _create_queue_and_ops(self):
        """
        Create a shuffled queue or FIFO queue, and create queue
        operations. This should be called before tf.Graph.finalize.
        """

        # create a queue
        if self.shuffle:
            self._queue = tf.RandomShuffleQueue(
                capacity=self.capacity,
                min_after_dequeue=self.min_queue_size,
                dtypes=[holder.dtype for holder in self.__placeholders],
                shapes=None,
                names=self.__fields,
                name="shuffled_queue")
        else:
            self._queue = tf.FIFOQueue(
                # pylint: disable=redefined-variable-type
                capacity=self.capacity,
                dtypes=[holder.dtype for holder in self.__placeholders],
                shapes=None,
                names=self.__fields,
                name="FIFO_queue")

        # create queue operations
        queue_input_dict = dict(zip(self.__fields, self.__placeholders))
        self._enqueue_op = self._queue.enqueue(queue_input_dict)
        self._dequeue_op = self._queue.dequeue_many
        self._query_queue_size_op = self._queue.size()
        self._close_queue_op = self._queue.close(cancel_pending_enqueues=True)

    def _push(self, thread_id):
        tf.logging.info('New thread: {}'.format(thread_id))
        try:
            print('push thread')
            #if self.shuffle:
            #    iterator = self.sampler()
            #else:
            #    iterator = self.sampler(self.batch_size)
            for image_window in self():
                if self._session._closed:
                    break
                if self._coordinator.should_stop():
                    break

            #    patch_dict = patch.as_dict(self._placeholders)
                self._session.run(self._enqueue_op, feed_dict=image_window)

            ## push a set of stopping patches
            #for i in range(0, self.capacity):
            #    if self._session._closed:
            #        break
            #    if self._coordinator.should_stop():
            #        break
            #    patch.fill_with_stopping_info()
            #    self._session.run(
            #        self._enqueue_op,
            #        feed_dict=patch.as_dict(self.sampler.placeholders))

        except tf.errors.CancelledError:
            pass
        #except ValueError as e:
        #    tf.logging.fatal(e)
        #    self.close_all()
        #except RuntimeError as e:
        #    tf.logging.fatal(e)
        #    self.close_all()
        except Exception as e:
            import sys
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(
                    exc_type, exc_value, exc_traceback, file=sys.stdout)
            self.close_all()
        finally:
            pass

    @property
    def current_queue_size(self):
        if self._session._closed:
            return 0
        return self._session.run(self._query_queue_size_op)

    def pop_batch_op(self, device_id=0):
        with tf.name_scope('pop_batch_{}'.format(device_id)):
            return self._dequeue_op(self.batch_size)

    def run_threads(self, session, coord, num_threads=1):
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
            #self._coordinator.join(threads=self._threads,
            #                       stop_grace_period_secs=0,
            #                       ignore_live_threads=True)
        except RuntimeError as e:
            tf.logging.info(e)
        finally:
            if not self._session._closed:
                self._session.run(self._close_queue_op)


class DeployInputBuffer(InputBatchQueueRunner):
    def __init__(self, batch_size, capacity, sampler):
        super(DeployInputBuffer, self).__init__(batch_size=batch_size,
                                                capacity=capacity,
                                                placeholders_dict=sampler,
                                                shuffle=False)


class TrainEvalInputBuffer(InputBatchQueueRunner):
    def __init__(self, batch_size, capacity, sampler, shuffle=True):
        super(TrainEvalInputBuffer, self).__init__(batch_size=batch_size,
                                                   capacity=capacity,
                                                   samplplaceholders_dicter=sampler,
                                                   shuffle=shuffle)
