import numpy as np
import tensorflow as tf

from layer.input_buffer import DeployInputBuffer, TrainEvalInputBuffer
from layer.input_sampler import ImageSampler


class InputQueueTest(tf.test.TestCase):
    def test_setup_train_eval_queue(self):
        test_sampler = ImageSampler(image_shape=(32, 32, 32),
                                    label_shape=(32, 32, 32),
                                    image_dtype=tf.float32,
                                    label_dtype=tf.int64,
                                    spatial_rank=3,
                                    num_modality=1,
                                    name='sampler_with_label')
        image_key, label_key, info_key = test_sampler.placeholder_names
        test_queue = TrainEvalInputBuffer(batch_size=2,
                                          capacity=8,
                                          sampler=test_sampler)
        out_1 = test_queue.pop_batch_op

        with self.test_session() as sess:
            coord = tf.train.Coordinator()
            test_queue.run_threads(sess, coord, num_threads=2)
            try:
                for i in range(100):
                    out_tuple = sess.run(out_1)
                    print image_key
                    print out_tuple[image_key].shape
            except tf.errors.OutOfRangeError as e:
                pass

    def test_deploy_queue(self):
        test_sampler = ImageSampler(image_shape=(32, 32, 32),
                                    label_shape=None,
                                    image_dtype=tf.float32,
                                    label_dtype=None,
                                    spatial_rank=3,
                                    num_modality=1,
                                    name='sampler_without_label')
        image_key, info_key = test_sampler.placeholder_names
        deploy_queue = DeployInputBuffer(batch_size=5,
                                         capacity=8,
                                         sampler=test_sampler)
        out_2 = deploy_queue.pop_batch_op
        with self.test_session() as sess:
            coord = tf.train.Coordinator()
            deploy_queue.run_threads(sess, coord, num_threads=1)
            try:
                for i in range(100):
                    out_tuple = sess.run(out_2)
                    print image_key
                    print out_tuple[image_key].shape
                    print out_tuple[info_key]
            except tf.errors.OutOfRangeError as e:
                pass


if __name__ == "__main__":
    tf.test.main()
