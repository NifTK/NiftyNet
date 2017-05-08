import numpy as np
import tensorflow as tf

from layer.highresblock import HighResBlock


class HighResBlockTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)

        highres_layer = HighResBlock(n_output_chns=8, kernels=(3, 3), with_res=True)
        out = highres_layer(x, is_training=True)
        print highres_layer.to_string()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)

if __name__ == "__main__":
    tf.test.main()
