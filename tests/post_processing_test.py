from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.post_processing import PostProcessingLayer
from tests.niftynet_testcase import NiftyNetTestCase

class PostProcessingTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def test_3d_shape(self):
        x = self.get_3d_input()
        post_process_layer = PostProcessingLayer("SOFTMAX")
        print(post_process_layer)
        out_post = post_process_layer(x)
        print(post_process_layer)

        with self.cached_session() as sess:
            out = sess.run(out_post)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)

    def test_2d_shape(self):
        x = self.get_2d_input()
        post_process_layer = PostProcessingLayer("IDENTITY")
        out_post = post_process_layer(x)
        print(post_process_layer)

        with self.cached_session() as sess:
            out = sess.run(out_post)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)

    def test_3d_argmax_shape(self):
        x = self.get_3d_input()
        post_process_layer = PostProcessingLayer("ARGMAX")
        out_post = post_process_layer(x)
        print(post_process_layer)

        with self.cached_session() as sess:
            out = sess.run(out_post)
            x_shape = tuple(x.shape.as_list()[:-1])
            self.assertAllClose(x_shape + (1,), out.shape)


if __name__ == "__main__":
    tf.test.main()
