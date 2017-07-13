from __future__ import absolute_import, print_function

import os
import sys

import tensorflow as tf

from niftynet.engine.input_buffer import DeployInputBuffer, TrainEvalInputBuffer
from niftynet.engine.toy_sampler import ToySampler
from niftynet.utilities.input_placeholders import ImagePatch

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')


class InputQueueTest(tf.test.TestCase):
    def test_3d_setup_train_eval_queue(self):
        test_patch = ImagePatch(spatial_rank=3,
                                image_size=32,
                                label_size=32,
                                weight_map_size=32,
                                image_dtype=tf.float32,
                                label_dtype=tf.int64,
                                weight_map_dtype=tf.float32,
                                num_image_modality=1,
                                num_label_modality=1,
                                num_weight_map=1)
        test_sampler = ToySampler(test_patch, name='sampler')
        image_key, label_key, info_key, weight_map_key = \
            test_sampler.placeholder_names
        test_queue = TrainEvalInputBuffer(batch_size=2,
                                          capacity=8,
                                          sampler=test_sampler)
        out_1 = test_queue.pop_batch_op

        with self.test_session() as sess:
            coord = tf.train.Coordinator()
            test_queue.run_threads(sess, coord, num_threads=2)
            try:
                for i in range(3):
                    out_tuple = sess.run(out_1)
                    if test_patch.is_stopping_signal(out_tuple[info_key][-1]):
                        test_queue.close_all()
                    self.assertAllClose(
                        (2, 32, 32, 32, 1), out_tuple[image_key].shape)
                test_queue.close_all()
            except tf.errors.OutOfRangeError as e:
                pass

    def test_3d_deploy_queue(self):
        test_patch = ImagePatch(spatial_rank=3,
                                image_size=32,
                                image_dtype=tf.float32,
                                num_image_modality=1)
        test_sampler = ToySampler(test_patch, name='sampler')
        image_key, info_key = test_sampler.placeholder_names
        deploy_queue = DeployInputBuffer(batch_size=5,
                                         capacity=8,
                                         sampler=test_sampler)
        out_2 = deploy_queue.pop_batch_op
        with self.test_session() as sess:
            coord = tf.train.Coordinator()
            deploy_queue.run_threads(sess, coord, num_threads=1)
            try:
                for i in range(3):
                    out_tuple = sess.run(out_2)
                    if test_patch.is_stopping_signal(out_tuple[info_key][-1]):
                        deploy_queue.close_all()
                    # print(out_tuple[info_key])
                    self.assertAllClose(
                        (5, 32, 32, 32, 1), out_tuple[image_key].shape)
                deploy_queue.close_all()
            except tf.errors.OutOfRangeError as e:
                pass

    def test_2d_setup_train_eval_queue(self):
        test_patch = ImagePatch(spatial_rank=2,
                                image_size=32,
                                label_size=32,
                                weight_map_size=32,
                                image_dtype=tf.float32,
                                label_dtype=tf.int64,
                                weight_map_dtype=tf.float32,
                                num_image_modality=1,
                                num_label_modality=1,
                                num_weight_map=1)
        test_sampler = ToySampler(test_patch, name='sampler')
        image_key, label_key, info_key, weight_map_key = \
            test_sampler.placeholder_names
        test_queue = TrainEvalInputBuffer(batch_size=2,
                                          capacity=8,
                                          sampler=test_sampler)
        out_1 = test_queue.pop_batch_op

        with self.test_session() as sess:
            coord = tf.train.Coordinator()
            test_queue.run_threads(sess, coord, num_threads=2)
            try:
                for i in range(3):
                    out_tuple = sess.run(out_1)
                    if test_patch.is_stopping_signal(out_tuple[info_key][-1]):
                        test_queue.close_all()
                    self.assertAllClose(
                        (2, 32, 32, 1), out_tuple[image_key].shape)
                test_queue.close_all()
            except tf.errors.OutOfRangeError as e:
                pass

    def test_2d_deploy_queue(self):
        test_patch = ImagePatch(spatial_rank=2,
                                image_size=32,
                                image_dtype=tf.float32,
                                num_image_modality=1)
        test_sampler = ToySampler(test_patch, name='sampler')
        image_key, info_key = test_sampler.placeholder_names
        deploy_queue = DeployInputBuffer(batch_size=5,
                                         capacity=8,
                                         sampler=test_sampler)
        out_2 = deploy_queue.pop_batch_op
        with self.test_session() as sess:
            coord = tf.train.Coordinator()
            deploy_queue.run_threads(sess, coord, num_threads=1)
            try:
                for i in range(20):
                    out_tuple = sess.run(out_2)
                    if test_patch.is_stopping_signal(out_tuple[info_key][-1]):
                        deploy_queue.close_all()
                    # print(out_tuple[info_key])
                    self.assertAllClose(
                        (5, 32, 32, 1), out_tuple[image_key].shape)
                deploy_queue.close_all()
            except tf.errors.OutOfRangeError as e:
                pass


if __name__ == "__main__":
    tf.test.main()
