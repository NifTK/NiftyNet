import time
import threading
import tensorflow as tf

class InputBatchQueueRunner(object):
    def __init__(self, batch_size, capacity,
                 input_names, input_types, input_shapes, generator, shuffle):
        self.batch_size = batch_size
        self.element_generator = generator
        self.threads = []

        self.queue = []
        self.place_holders = []
        self.enqueue_op = []
        self.close_queue_op = []
        self.queue_size_op = []
        self.init_queue(
                input_names, input_types, input_shapes, capacity, shuffle)

    def init_queue(self, input_names, input_types, input_shapes,
                   capacity, shuffle):
        # create queue
        if shuffle:
            self.queue = tf.RandomShuffleQueue(
                capacity=capacity, min_after_dequeue=capacity/2,
                dtypes=input_types, shapes=input_shapes, names=input_names,
                name="shuffled_queue")
        else:
            self.queue = tf.FIFOQueue(
                capacity=capacity,
                dtypes=input_types, shapes=input_shapes, names=input_names,
                name="FIFO_queue")
        # create place holders
        self.place_holders = tuple(tf.placeholder(dtype,
                                                  shape=input_shapes[i],
                                                  name=input_names[i])
                                   for i, dtype in enumerate(input_types))
        # create enqueue operation
        queue_input_dict = dict(zip(input_names, self.place_holders))
        self.enqueue_op = self.queue.enqueue(queue_input_dict)
        self.close_queue_op = self.queue.close(cancel_pending_enqueues=True)
        self.queue_size_op = self.queue.size()

    def push(self, session, coord):
        try:
            for t in self.element_generator():
                if coord.should_stop():
                    break
                session.run(self.enqueue_op, feed_dict={self.place_holders:t})

            retry = 0
            while retry < 5:
                if self.batch_size <= session.run(self.queue_size_op):
                    # there are batches to be processed before delete the queue
                    time.sleep(2)
                    retry = retry + 1
                else:
                    break
        except tf.errors.CancelledError:
            pass
        finally:
            if self.batch_size > session.run(self.queue_size_op):
                print "Insufficient samples to form a batch:"\
                    "remaining {} in queue".format(
                            session.run(self.queue_size_op))
            self.close_all(coord, session)
            print 'preprocessing threads finished'

    def pop(self, n):
        return self.queue.dequeue_many(n)

    def pop_batch(self):
        return self.pop(self.batch_size)

    def init_threads(self, session, coord, num_threads):
        print 'Starting preprocessing threads...'
        for i in xrange(num_threads):
            self.threads.append(
                threading.Thread(target=self.push, args=(session, coord)))
            self.threads[i].daemon = True
            self.threads[i].start()

    def close_all(self, coord, session):
        try:
            coord.request_stop()
            session.run(self.close_queue_op)
        except Exception as e:
            print e


class DeployInputBuffer(InputBatchQueueRunner):
    def __init__(self, batch_size, capacity, shapes, sample_generator):
        input_names = ("images", "info")
        input_types = (tf.float32, tf.int64)
        super(DeployInputBuffer, self).__init__(
            batch_size, capacity,
            input_names, input_types, shapes, sample_generator, False)


class TrainEvalInputBuffer(InputBatchQueueRunner):
    def __init__(self, batch_size, capacity, shapes,
                 sample_generator, shuffle=True):
        input_names = ("images", "labels", "info")
        input_types = (tf.float32, tf.int64, tf.int64)
        super(TrainEvalInputBuffer, self).__init__(
            batch_size, capacity,
            input_names, input_types, shapes, sample_generator, shuffle)
