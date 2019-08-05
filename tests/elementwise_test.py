from __future__ import absolute_import, print_function
import tensorflow as tf

from niftynet.layer.elementwise import ElementwiseLayer


class ElementwiseTest(tf.test.TestCase):
    def test_3d_shape(self):
        input_shape = (2, 16, 16, 16, 6)
        x_1 = tf.ones(input_shape)
        input_shape = (2, 16, 16, 16, 6)
        x_2 = tf.zeros(input_shape)
        sum_layer = ElementwiseLayer('SUM')
        out_sum_1 = sum_layer(x_1, x_2)

        input_shape = (2, 16, 16, 16, 8)
        x_1 = tf.ones(input_shape)
        input_shape = (2, 16, 16, 16, 6)
        x_2 = tf.zeros(input_shape)
        sum_layer = ElementwiseLayer('SUM')
        out_sum_2 = sum_layer(x_1, x_2)

        input_shape = (2, 16, 16, 16, 6)
        x_1 = tf.ones(input_shape)
        input_shape = (2, 16, 16, 16, 8)
        x_2 = tf.zeros(input_shape)
        sum_layer = ElementwiseLayer('SUM')
        out_sum_3 = sum_layer(x_1, x_2)

        input_shape = (2, 16, 16, 16, 6)
        x_1 = tf.ones(input_shape)
        input_shape = (2, 16, 16, 16, 8)
        x_2 = tf.zeros(input_shape)
        sum_layer = ElementwiseLayer('CONCAT')
        out_sum_4 = sum_layer(x_1, x_2)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_sum_1)
            self.assertAllClose((2, 16, 16, 16, 6), out.shape)
            out = sess.run(out_sum_2)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)
            out = sess.run(out_sum_3)
            self.assertAllClose((2, 16, 16, 16, 6), out.shape)
            out = sess.run(out_sum_4)
            self.assertAllClose((2, 16, 16, 16, 14), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 16, 16, 6)
        x_1 = tf.ones(input_shape)
        input_shape = (2, 16, 16, 6)
        x_2 = tf.zeros(input_shape)
        sum_layer = ElementwiseLayer('SUM')
        out_sum_1 = sum_layer(x_1, x_2)

        input_shape = (2, 16, 16, 8)
        x_1 = tf.ones(input_shape)
        input_shape = (2, 16, 16, 6)
        x_2 = tf.zeros(input_shape)
        sum_layer = ElementwiseLayer('SUM')
        out_sum_2 = sum_layer(x_1, x_2)

        input_shape = (2, 16, 16, 6)
        x_1 = tf.ones(input_shape)
        input_shape = (2, 16, 16, 8)
        x_2 = tf.zeros(input_shape)
        sum_layer = ElementwiseLayer('SUM')
        out_sum_3 = sum_layer(x_1, x_2)

        input_shape = (2, 16, 16, 6)
        x_1 = tf.ones(input_shape)
        input_shape = (2, 16, 16, 8)
        x_2 = tf.zeros(input_shape)
        sum_layer = ElementwiseLayer('CONCAT')
        out_sum_4 = sum_layer(x_1, x_2)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_sum_1)
            self.assertAllClose((2, 16, 16, 6), out.shape)
            out = sess.run(out_sum_2)
            self.assertAllClose((2, 16, 16, 8), out.shape)
            out = sess.run(out_sum_3)
            self.assertAllClose((2, 16, 16, 6), out.shape)
            out = sess.run(out_sum_4)
            self.assertAllClose((2, 16, 16, 14), out.shape)


if __name__ == "__main__":
    tf.test.main()
