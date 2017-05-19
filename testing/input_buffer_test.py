import tensorflow as tf

from layer.input_buffer import DeployInputBuffer, TrainEvalInputBuffer
from layer.input_placeholders import ImagePatch
from layer.toy_sampler import ToySampler


class InputQueueTest(tf.test.TestCase):
    def test_3d_setup_train_eval_queue(self):
        test_patch = ImagePatch(image_shape=(32, 32, 32),
                                label_shape=(32, 32, 32),
                                weight_map_shape=(32, 32, 32),
                                image_dtype=tf.float32,
                                label_dtype=tf.int64,
                                weight_map_dtype=tf.float32,
                                num_modality=1,
                                num_map=1,
                                name='image_patch')
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
                for i in range(100):
                    out_tuple = sess.run(out_1)
                    #print image_key
                    self.assertAllClose(
                        (2, 32, 32, 32, 1), out_tuple[image_key].shape)
            except tf.errors.OutOfRangeError as e:
                pass

    def test_3d_deploy_queue(self):
        test_patch = ImagePatch(image_shape=(32, 32, 32),
                                image_dtype=tf.float32,
                                num_modality=1,
                                name='sampler_without_label')
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
                for i in range(100):
                    out_tuple = sess.run(out_2)
                    #print out_tuple[info_key]
                    self.assertAllClose(
                        (5, 32, 32, 32, 1), out_tuple[image_key].shape)
            except tf.errors.OutOfRangeError as e:
                pass

    def test_2d_setup_train_eval_queue(self):
        test_patch = ImagePatch(image_shape=(32, 32),
                                label_shape=(32, 32),
                                weight_map_shape=(32, 32),
                                image_dtype=tf.float32,
                                label_dtype=tf.int64,
                                weight_map_dtype=tf.float32,
                                num_modality=1,
                                num_map=1,
                                name='image_patch')
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
                for i in range(100):
                    out_tuple = sess.run(out_1)
                    #print image_key
                    self.assertAllClose(
                        (2, 32, 32, 1), out_tuple[image_key].shape)
            except tf.errors.OutOfRangeError as e:
                pass

    def test_2d_deploy_queue(self):
        test_patch = ImagePatch(image_shape=(32, 32),
                                image_dtype=tf.float32,
                                num_modality=1,
                                name='sampler_without_label')
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
                for i in range(100):
                    out_tuple = sess.run(out_2)
                    #print out_tuple[info_key]
                    self.assertAllClose(
                        (5, 32, 32, 1), out_tuple[image_key].shape)
            except tf.errors.OutOfRangeError as e:
                pass

if __name__ == "__main__":
    tf.test.main()
