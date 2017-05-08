import numpy as np
import tensorflow as tf

from downsample import DownSampleLayer


class DownSampleTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)

        down_sample_layer = DownSampleLayer('MAX', 3, 3)
        out_down_sample = down_sample_layer(x)
        print down_sample_layer.to_string()
        print out_down_sample.get_shape()

        down_sample_layer = DownSampleLayer('AVG', 2, 2)
        out_down_sample = down_sample_layer(x)
        print down_sample_layer.to_string()
        print out_down_sample.get_shape()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_down_sample)
            #self.assertAllClose(input_shape, out.shape)
            #self.assertAllClose(np.zeros(input_shape), out)

if __name__ == "__main__":
    tf.test.main()
