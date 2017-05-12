import numpy as np
import tensorflow as tf

from nn.input_queue import InputBatchQueueRunner, DeployInputBuffer, TrainEvalInputBuffer


class InputQueueTest(tf.test.TestCase):

    def get_dummy_generator(self, num_samples=10, window_size=10, with_seg=True):
        def dummy_generator():
            for i in range(num_samples):
                image = np.ones([window_size]*3) * 1.0
                info = np.zeros(7)
                if with_seg:
                    label = np.ones([window_size]*3) * 1
                    yield image, label, info
                else:
                    yield image, info
        return dummy_generator

    def test_setup_train_eval_queue(self):
        image_shape = 20
        image_shapes = [image_shape] * 3
        label_shapes = [image_shape] * 3
        input_shapes = [image_shapes, label_shapes, [7]]
        generator = self.get_dummy_generator(num_samples=10,
                                             window_size=image_shape,
                                             with_seg=True)
        test_queue = TrainEvalInputBuffer(batch_size=2,
                                          capacity=8,
                                          shapes=input_shapes,
                                          sample_generator=generator)
        image_shape = 20
        image_shapes = [image_shape] * 3
        input_shapes = [image_shapes, [7]]
        generator = self.get_dummy_generator(num_samples=10,
                                             window_size=image_shape,
                                             with_seg=False)
        deploy_queue = DeployInputBuffer(batch_size=2,
                                         capacity=8,
                                         shapes=input_shapes,
                                         sample_generator=generator)
        out_1 = test_queue.pop_batch_op
        out_2 = deploy_queue.pop_batch_op

        with self.test_session() as sess:
            coord = tf.train.Coordinator()
            test_queue.run_threads(sess, coord, num_threads=2)
            deploy_queue.run_threads(sess, coord, num_threads=2)
            try:
                for i in range(100):
                    out_tuple = sess.run(out_1)
                    print out_tuple['images'].shape
                    out_tuple = sess.run(out_2)
                    print i, out_tuple['images'].shape
            except tf.errors.OutOfRangeError as e:
                pass

if __name__ == "__main__":
    tf.test.main()
