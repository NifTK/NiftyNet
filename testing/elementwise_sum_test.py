import numpy as np
import tensorflow as tf

from layer.elementwise_sum import ElementwiseSumLayer


class ElementwiseSumTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 16, 16, 16, 6)
        x_1 = tf.ones(input_shape)
        input_shape = (2, 16, 16, 16, 6)
        x_2 = tf.zeros(input_shape)
        sum_layer = ElementwiseSumLayer()
        out_sum = sum_layer(x_1, x_2)
        print sum_layer.to_string()

        input_shape = (2, 16, 16, 16, 8)
        x_1 = tf.ones(input_shape)
        input_shape = (2, 16, 16, 16, 6)
        x_2 = tf.zeros(input_shape)
        sum_layer = ElementwiseSumLayer()
        out_sum = sum_layer(x_1, x_2)
        print sum_layer.to_string()

        input_shape = (2, 16, 16, 16, 6)
        x_1 = tf.ones(input_shape)
        input_shape = (2, 16, 16, 16, 8)
        x_2 = tf.zeros(input_shape)
        sum_layer = ElementwiseSumLayer()
        out_sum = sum_layer(x_1, x_2)
        print sum_layer.to_string()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_sum)

if __name__ == "__main__":
    tf.test.main()
